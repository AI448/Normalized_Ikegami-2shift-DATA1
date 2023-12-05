from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Set, Tuple
from pandas import Timestamp as Date
from datetime import timedelta
from math import inf
import pulp
import itertools

import sys
import pandas as pd


@dataclass(frozen=True)
class Group:
    """グループ"""
    id: str


@dataclass(frozen=True)
class Member:
    """メンバー"""
    id: str


class Shift(Enum):
    """シフト種別"""
    OFF = "OFF"
    DAY = "DAY"
    NIGHT = "NIGHT"
    MORNING = "MORNING"
    OTHER = "OTHER"


@dataclass(frozen=True)
class NGPattern:
    """NGパターンのキー"""
    id: str


def _date_to_id(date: Date):
    return date.strftime("%Y%m%d")


class Ikegami2ShiftModel:
    """最適化モデル"""

    # MEMO: モデルを修正して何度も実行したりするときにはこのようなクラスを作ったほうが便利なこともあるが，
    # 今回の簡単な例ではあまり意味はなく，データに対してモデルを返す関数
    #   def build_model(...) -> pulp.LpProblem
    # または，データに対して割当結果を返す関数
    #   def solve(...) -> Dict[Tuple[Member, Date], Shift]
    # でも十分．

    def __init__(
        self,
        members: Set[Member],
        dates: List[Date],
        groups: Set[Group],
        group_members: Set[Tuple[Group, Member]],
        ng_petterns: Dict[Tuple[NGPattern, int], Shift],
        consecutive_shifts_lower: Dict[Shift, int],
        consecutive_shifts_upper: Dict[Shift, int],
        days_between_shifts_lower: Dict[Shift, int],
        days_between_shifts_upper: Dict[Shift, int],
        assignments_lower: Dict[Tuple[Member, Shift], int],
        assignments_upper: Dict[Tuple[Member, Shift], int],
        weekends_off_lower: Dict[Member, int],
        weekends_off_upper: Dict[Member, int],
        number_of_workers_lower: Dict[Tuple[Group, Date], int],
        number_of_workers_upper: Dict[Tuple[Group, Date], int],
        request_shift: Dict[Tuple[Member, Date], Shift],
        refused_shift: Dict[Tuple[Member, Date], Shift]
    ):
        """データから最適化モデルを構築する"""

        saturdays = {date for date in dates if date.day_of_week == 5}

        group_to_members: Dict[Group, List[Member]] = {}
        for group, member in group_members:
            group_to_members.setdefault(group, []).append(member)

        ng_pattern_to_order_shift: Dict[NGPattern, List[Tuple[int, Shift]]] = {}
        for (ng_pattern, order), shift in ng_petterns.items():
            ng_pattern_to_order_shift.setdefault(ng_pattern, []).append((order, shift))

        self.problem = pulp.LpProblem()

        #
        # 変数の定義(今回は変数が 2 種類だけなので x と y にした)
        #
        # member が date に shift を実行するなら 1 そうでなければ 0
        self.x: Dict[Tuple[Member, date, Shift], pulp.LpVariable] = {}
        for member, date, shift in itertools.product(members, dates, Shift):
            self.x[member, date, shift] = pulp.LpVariable(
                name=f"x_{member.id}_{_date_to_id(date)}_{shift.value}",
                cat=pulp.LpBinary
            )
        # member が date (土曜日)から 2 連休を取るなら 1 そうでなければ 0
        self.y: Dict[Tuple[member, date], pulp.LpVariable] = {}
        for member, date in itertools.product(members, saturdays):
            self.y[member, date] = pulp.LpVariable(
                name=f"y_{member.id}_{_date_to_id(date)}",
                cat=pulp.LpBinary
            )

        #
        # 制約条件の定義
        #

        # 各 member は各 date においてちょうど 1 つの shift を実行する
        for member, date in itertools.product(members, dates):
            self.problem += (
                pulp.lpSum(self.x[member, date, shift] for shift in Shift) == 1,
                f"c_assignment_{member.id}_{_date_to_id(date)}"
            )

        # member, date において OTHER がリクエストされていない場合には OTHER の勤務を実行できない
        for member, date in itertools.product(members, dates):
            if request_shift.get((member, date), None) != Shift.OTHER:
                self.problem += (
                    self.x[member, date, Shift.OTHER] == 0,
                    f"c_restrict_OTHER_{member.id}_{_date_to_id(date)}"
                )

        # 夜勤
        for member, date in itertools.product(members, dates[:-1]):
            # date に夜勤 ⇔ date + 1 に夜勤明け
            self.problem += (
                self.x[member, date, Shift.NIGHT] == self.x[member, date + timedelta(1), Shift.MORNING],
                f"c_night_rule_{member.id}_{_date_to_id(date)}"
            )

        # NG パターン
        for member, (ng_pattern, order_shifts) in itertools.product(members, ng_pattern_to_order_shift.items()):
            assert len(order_shifts) >= 1
            assert min(order_shifts)[0] == 0  # 0 始まりを仮定している
            for date in dates[:-max(order_shifts)[0]]:
                # order_shifts に合致する勤務のうち少なくとも 1 つは 0
                self.problem +=(
                    pulp.lpSum(self.x[member, date + timedelta(d), shift] for d, shift in order_shifts) <= len(order_shifts) - 1,
                    f"c_ng_{member.id}_{ng_pattern.id}_{_date_to_id(date)}"
                )

        # 勤務人数の制約
        for group, date, shift in itertools.product(groups, dates, Shift):
            if number_of_workers_lower.get((group, date, shift), -inf) > 0:
                # 勤務人数合計 >= 勤務人数下限
                self.problem += (
                    pulp.lpSum(self.x[member, date, shift] for member in group_to_members[group]) >= number_of_workers_lower[group, date, shift],
                    f"c_number_of_workers_lower_{group.id}_{_date_to_id(date)}_{shift.value}"
                )
            if number_of_workers_upper.get((group, date, shift), inf) < inf:
                # 勤務人数合計 <= 勤務人数上限
                self.problem += (
                    pulp.lpSum(self.x[member, date, shift] for member in group_to_members[group]) <= number_of_workers_upper[group, date, shift],
                    f"c_number_of_workers_upper_{group.id}_{_date_to_id(date)}_{shift.value}"
                )

        # 勤務回数の制約
        for member, shift in itertools.product(members, Shift):
            # 勤務回数 >= 勤務回数下限
            if assignments_lower.get((member, shift), -inf) > 0:
                self.problem += (
                    pulp.lpSum(self.x[member, date, shift] for date in dates) >= assignments_lower[member, shift],
                    f"c_number_of_working_lower_{member.id}_{shift.value}"
                )
            # 勤務回数 <= 勤務回数上限
            if assignments_upper.get((member, shift), inf) < inf:
                self.problem += (
                    pulp.lpSum(self.x[member, date, shift] for date in dates) <= assignments_upper[member, shift],
                    f"c_number_of_working_upper_{member.id}_{shift.value}"
                )
            # 連続勤務回数下限
            if consecutive_shifts_lower.get(shift, -inf) >= 2:
                for date in dates[1:]:
                    for d in range(1, consecutive_shifts_lower[shift]):
                        if date + timedelta(d) not in dates:
                            continue
                        # date-1 にそのシフトの勤務をしないかつdateにそのシフトの勤務を行う ⇒ date+d についてもそのシフトの勤務を行う
                        self.problem += (
                            -self.x[member, date - timedelta(1), shift] + self.x[member, date, shift] <= self.x[member, date + timedelta(d), shift],
                            f"c_consecutive_shifts_lower_{member.id}_{shift.value}_{_date_to_id(date)}_{d}"
                        )
            # 連続勤務回数上限
            if consecutive_shifts_upper.get(shift, inf) < inf:
                for date in dates[:-consecutive_shifts_upper[shift]]:
                    # 連続する consecutive_shifts_upper[shift]+1 日間での当該シフトの勤務回数が consecutive_shifts_upper[shift] 以下
                    self.problem += (
                        pulp.lpSum(self.x[member, date + timedelta(d), shift] for d in range(consecutive_shifts_upper[shift] + 1))
                        <= consecutive_shifts_upper[shift],
                        f"c_consecutive_shifts_upper_{member.id}_{shift.value}_{_date_to_id(date)}"
                    )
            # 勤務間隔下限
            if days_between_shifts_lower.get(shift, -inf) >= 2:
                for date in dates[:-days_between_shifts_lower[shift] - 1]:
                    # 連続する days_between_shifts_lower[shift] 日間での当該シフトの勤務回数が 1 以下
                    self.problem += (
                        pulp.lpSum(self.x[member, date + timedelta(d), shift] for d in range(days_between_shifts_lower[shift])) <= 1,
                        f"c_days_between_shifts_lower_{member.id}_{shift.value}_{_date_to_id(date)}"
                    )
            # 勤務間隔上限
            if days_between_shifts_upper.get(shift, inf) < inf:
                for date in dates[:-days_between_shifts_upper[shift]-1]:
                    # 連続する days_between_shifts_upper[shift]+1 日間での当該シフトの勤務回数が 1 以上
                    self.problem += (
                        pulp.lpSum(self.x[member, date + timedelta(d), shift] for d in range(days_between_shifts_upper[shift]+1)) >= 1,
                        f"c_days_between_shifts_upper_{member.id}_{shift.value}_{_date_to_id(date)}"
                    )

        for member in members:
            # x と y の関係
            for date in saturdays:
                if date + timedelta(1) not in dates:
                    # date の翌日の日曜日が翌月になる場合には 2 連休とみなさないことにする
                    self.problem += (
                        self.y[member, date] == 0
                    )
                else:
                    # date において y が 1 ⇒ date と date+1 において x_OFF が 1
                    self.problem += (
                        self.y[member, date] <= self.x[member, date, Shift.OFF],
                        f"c_yx_SAT_{member.id}_{_date_to_id(date)}"
                    )
                    self.problem += (
                        self.y[member, date] <= self.x[member, date + timedelta(1), Shift.OFF],
                        f"c_yx_SUN_{member.id}_{_date_to_id(date)}"
                    )
                    # date と date+1 において x_OFF が 1 ⇒ date において y が 1
                    self.problem += (
                        self.x[member, date, Shift.OFF] + self.x[member, date + timedelta(1), Shift.OFF] - 1 <= self.y[member, date],
                        f"c_yx_{member.id}_{_date_to_id(date)}"
                    )

            # 週末の 2 連休回数下限
            if weekends_off_lower.get(member, -inf) > 0:
                self.problem += (
                    pulp.lpSum(self.y[member, date] for date in saturdays) >= weekends_off_lower[member],
                    f"c_weekends_off_lower_{member.id}_{_date_to_id(date)}"
                )
            # 週末の 2 連休回数上限
            if weekends_off_upper.get(member, inf) < inf:
                self.problem += (
                    pulp.lpSum(self.y[member, date] for date in saturdays) <= weekends_off_lower[member],
                    f"c_weekends_off_upper_{member.id}_{_date_to_id(date)}"
                )

        for (member, date), shift in refused_shift.items():
            # 割当禁止
            self.problem += (
                self.x[member, date, shift] == 0,
                f"c_refused_shift_{member.id}_{_date_to_id(date)}_{shift.value}"
            )

        #
        # 目的関数の定義
        #

        # リクエスト通りの勤務が割当たらなかった数を最小化
        self.problem += pulp.lpSum(1 - self.x[member, date, shift] for (member, date), shift in request_shift.items())

        # 終わり


    def solve(self, time_limit=60) -> Dict[Tuple[Member, Date], Shift]:
        """最適化問題を解いてメンバー・日付ごとの割り当たったシフトを返す"""

        status = self.problem.solve(pulp.PULP_CBC_CMD(timeLimit=time_limit))
        assert status == 1

        member_date_to_shift = {}
        for member, date, shift in self.x:
            # NOTE: ソルバーによっては整数変数の値を整数値で返してくるとは限らない(誤差を含んだ値の可能性がある)ので 0.5 を閾値として判定している
            if pulp.value(self.x[member, date, shift]) > 0.5:
                assert (member, date) not in member_date_to_shift
                member_date_to_shift[member, date] = shift
        return member_date_to_shift


    def write_MPS(self, filepath: str) -> None:
        self.problem.writeMPS(filepath)


    def write_LP(self, filepath: str) -> None:
        self.problem.writeLP(filepath)


def main():

    # xlsx の読み込み（データの重複や型，値の範囲等のチェックはしていない）
    data_filepath = sys.argv[1]
    df_shift = pd.read_excel(data_filepath, "shift", dtype=str)
    df_ng_pettern = pd.read_excel(data_filepath, "ng_pattern", dtype=str)
    df_member = pd.read_excel(data_filepath, "member", dtype=str)
    df_member_shift = pd.read_excel(data_filepath, "member_shift", dtype=str)
    df_group_member = pd.read_excel(data_filepath, "group_member", dtype=str)
    df_group_date_shift = pd.read_excel(data_filepath, "group_date_shift", dtype=str)
    df_request_shift = pd.read_excel(data_filepath, "request_shift", dtype=str)
    df_refused_shift = pd.read_excel(data_filepath, "refused_shift", dtype=str)

    # データの構築

    # グループの集合
    groups = set(map(Group, df_group_member["group"]))
    # メンバーの集合
    members = set(map(Member, df_member["member"]))
    # 日付の集合
    dates = sorted(set(map(Date, df_group_date_shift["date"]))) # TODO
    # グループとメンバーの組の集合
    group_members = set(map(lambda x: (Group(x[0]), Member(x[1])), zip(df_group_member["group"], df_group_member["member"])))
    # NG パターン
    ng_petterns = {
        (NGPattern(ng_pattern), int(order)): Shift(shift)
        for ng_pattern, order, shift in zip(df_ng_pettern["ng_pattern"], df_ng_pettern["order"], df_ng_pettern["shift"])
    }
    # 連続割当回数下限
    consecutive_shifts_lower = {
        Shift(shift): int(lower)
        for shift, lower in zip(df_shift["shift"], df_shift["consecutive_shifts_lower"])
        if float(lower) > -inf  # NOTE NULLの除外
    }
    # 連続割当回数上限
    consecutive_shifts_upper = {
        Shift(shift): int(upper)
        for shift, upper in zip(df_shift["shift"], df_shift["consecutive_shifts_upper"])
        if float(upper) < inf
    }
    # 割当間隔下限
    days_between_shifts_lower = {
        Shift(shift): int(lower)
        for shift, lower in zip(df_shift["shift"], df_shift["days_between_shifts_lower"])
        if float(lower) > -inf
    }
    # 割当間隔上限
    days_between_shifts_upper = {
        Shift(shift): int(upper)
        for shift, upper in zip(df_shift["shift"], df_shift["days_between_shifts_upper"])
        if float(upper) < inf
    }
    # 割当回数下限
    number_of_working_lower = {
        (Member(member), Shift(shift)): int(lower)
        for member, shift, lower in zip(df_member_shift["member"], df_member_shift["shift"], df_member_shift["number_of_working_lower"])
        if float(lower) > -inf
    }
    # 割当回数上限
    number_of_working_upper = {
        (Member(member), Shift(shift)): int(upper)
        for member, shift, upper in zip(df_member_shift["member"], df_member_shift["shift"], df_member_shift["number_of_working_upper"])
        if float(upper) < inf
    }
    # 割当人数下限
    number_of_workers_lower = {
        (Group(group), Date(date), Shift(shift)): int(lower)
        for group, date, shift, lower in zip(df_group_date_shift["group"], df_group_date_shift["date"], df_group_date_shift["shift"], df_group_date_shift["number_of_workers_lower"])
        if float(lower) > -inf
    }
    # 割当人数上限
    number_of_workers_upper = {
        (Group(group), Date(date), Shift(shift)): int(upper)
        for group, date, shift, upper in zip(df_group_date_shift["group"], df_group_date_shift["date"], df_group_date_shift["shift"], df_group_date_shift["number_of_workers_upper"])
        if float(upper) < inf
    }
    # 週末 2 連休回数下限
    weekends_off_lower = {
        Member(member): int(lower)
        for member, lower in zip(df_member["member"], df_member["weekends_off_lower"])
        if float(lower) > -inf
    }
    # 週末 2 連休回数上限
    weekends_off_upper = {
        Member(member): int(upper)
        for member, upper in zip(df_member["member"], df_member["weekends_off_upper"])
        if float(upper) < inf
    }
    # 割当リクエスト
    request_shift = {
        (Member(member), Date(date)): Shift(shift)
        for member, date, shift in zip(df_request_shift["member"], df_request_shift["date"], df_request_shift["request_shift"])
    }
    # 割当禁止
    refused_shift = {
        (Member(member), Date(date)): Shift(shift)
        for member, date, shift in zip(df_refused_shift["member"], df_refused_shift["date"], df_refused_shift["shift"])
    }

    # モデルを構築
    model = Ikegami2ShiftModel(
        members=members,
        dates=dates,
        groups=groups,
        group_members=group_members,
        ng_petterns=ng_petterns,
        consecutive_shifts_lower=consecutive_shifts_lower,
        consecutive_shifts_upper=consecutive_shifts_upper,
        days_between_shifts_lower=days_between_shifts_lower,
        days_between_shifts_upper=days_between_shifts_upper,
        assignments_lower=number_of_working_lower,
        assignments_upper=number_of_working_upper,
        weekends_off_lower=weekends_off_lower,
        weekends_off_upper=weekends_off_upper,
        number_of_workers_lower=number_of_workers_lower,
        number_of_workers_upper=number_of_workers_upper,
        request_shift=request_shift,
        refused_shift=refused_shift
    )

    # LP ファイルを出力してみる
    model.write_LP("ikegami-2shift.lp")

    # MPS ファイルも出力してみる
    model.write_MPS("ikegami-2shift.mps")

    # 求解
    member_date_to_shift = model.solve()

    # 標準出力に結果を表示
    for date in dates:
        sys.stdout.write(f"\t{date.strftime('%m/%d')}")
    sys.stdout.write("\n")
    for member in members:
        sys.stdout.write(member.id)
        for date in dates:
            sys.stdout.write(f"\t{member_date_to_shift[member, date].value}")
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
