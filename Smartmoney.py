import numpy as np
import pandas as pd
from functools import wraps
from pandas import DataFrame, Series

# Декоратор для проверки входных данных DataFrame
def inputvalidator(func):
    @wraps(func)
    def wrap(cls, ohlc, *args, **kwargs):
        if not isinstance(ohlc, DataFrame):
            raise TypeError(f"Expected pandas.DataFrame, got {type(ohlc).__name__}")
        ohlc = ohlc.rename(columns={c: c.lower() for c in ohlc.columns})
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in ohlc.columns]
        if missing_columns:
            raise LookupError(f"DataFrame must have columns: {', '.join(missing_columns)}")
        return func(cls, ohlc, *args, **kwargs)
    return wrap

import pandas as pd
import numpy as np

class smc:
    @classmethod
    def fvg(cls, ohlc: pd.DataFrame) -> pd.DataFrame:
        fvg = np.where(
            ((ohlc["high"].shift(1) < ohlc["low"].shift(-1)) & (ohlc["close"] > ohlc["open"])) |
            ((ohlc["low"].shift(1) > ohlc["high"].shift(-1)) & (ohlc["close"] < ohlc["open"])),
            np.where(ohlc["close"] > ohlc["open"], 1, -1),
            np.nan,
        )

        top = np.where(
            ~np.isnan(fvg),
            np.where(ohlc["close"] > ohlc["open"], ohlc["low"].shift(-1), ohlc["low"].shift(1)),
            np.nan,
        )

        bottom = np.where(
            ~np.isnan(fvg),
            np.where(ohlc["close"] > ohlc["open"], ohlc["high"].shift(1), ohlc["high"].shift(-1)),
            np.nan,
        )

        mitigated_index = np.zeros(len(ohlc), dtype=np.int32)
        for i in np.where(~np.isnan(fvg))[0]:
            mask = np.zeros(len(ohlc), dtype=bool)
            if fvg[i] == 1:
                mask = ohlc["low"][i + 2:] <= top[i]
            elif fvg[i] == -1:
                mask = ohlc["high"][i + 2:] >= bottom[i]
            if np.any(mask):
                j = np.argmax(mask) + i + 2
                mitigated_index[i] = j

        mitigated_index = np.where(np.isnan(fvg), np.nan, mitigated_index)

        result = pd.concat(
            [
                pd.Series(fvg, index=ohlc.index, name="FVG"),
                pd.Series(top, index=ohlc.index, name="Top"),
                pd.Series(bottom, index=ohlc.index, name="Bottom"),
                pd.Series(mitigated_index, index=ohlc.index, name="MitigatedIndex"),
            ],
            axis=1,
        )

        return result

# Пример данных для DataFrame
data = {
    "open": [100, 105, 103, 108, 107],
    "high": [110, 108, 106, 111, 109],
    "low": [95, 102, 101, 105, 106],
    "close": [105, 103, 105, 110, 108]
}

df = pd.DataFrame(data)

result_df = smc.fvg(df)
print(result_df)



@classmethod
def swing_highs_lows(
    cls, ohlc: DataFrame, swing_length: int = 50
) -> Series:
    """
    Swing Highs and Lows
    A swing high is when the current high is the highest high out of the swing_length amount of candles before and after.
    A swing low is when the current low is the lowest low out of the swing_length amount of candles before and after.

    parameters:
    swing_length: int - the amount of candles to look back and forward to determine the swing high or low

    returns:
    HighLow = 1 if swing high, -1 if swing low
    Level = the level of the swing high or low
    """

    swing_length *= 2
    # set the highs to 1 if the current high is the highest high in the last 5 candles and next 5 candles
    swing_highs_lows = np.where(
        ohlc["high"]
        == ohlc["high"]
        .shift(-(swing_length // 2))
        .rolling(swing_length)
        .max(),
        1,
        np.where(
            ohlc["low"]
            == ohlc["low"]
            .shift(-(swing_length // 2))
            .rolling(swing_length)
            .min(),
            -1,
            np.nan,
        ),
    )

    continue_ = True
    while continue_:
        positions = np.where(~np.isnan(swing_highs_lows))[0]
        continue_ = False
        for i in range(len(positions) - 1):
            current, next = (
                swing_highs_lows[positions[i]],
                swing_highs_lows[positions[i + 1]],
            )
            high, low = (
                ohlc["high"].iloc[positions[i]],
                ohlc["low"].iloc[positions[i]],
            )
            next_high, next_low = (
                ohlc["high"].iloc[positions[i + 1]],
                ohlc["low"].iloc[positions[i + 1]],
            )
            if current == -1 and next == -1:
                remove_index = positions[i] if low > next_low else positions[i + 1]
                swing_highs_lows[remove_index] = np.nan
                continue_ = True
            elif current == 1 and next == 1:
                remove_index = (
                    positions[i] if high < next_high else positions[i + 1]
                )
                swing_highs_lows[remove_index] = np.nan
                continue_ = True

    positions = np.where(~np.isnan(swing_highs_lows))[0]
    if swing_highs_lows[positions[0]] == 1:
        swing_highs_lows[0] = -1
    if swing_highs_lows[positions[-1]] == -1:
        swing_highs_lows[-1] = 1

    level = np.where(
        ~np.isnan(swing_highs_lows),
        np.where(swing_highs_lows == 1, ohlc["high"], ohlc["low"]),
        np.nan,
    )

    return pd.concat(
        [
            pd.Series(swing_highs_lows, name="HighLow"),
            pd.Series(level, name="Level"),
        ],
        axis=1,
    )


@classmethod
def bos_choch(cls, ohlc: DataFrame, swing_highs_lows: DataFrame, close_break:bool = True) -> Series:
    """
    BOS - Break of Structure
    CHoCH - Change of Character
    these are both indications of market structure changing

    parameters:
    swing_highs_lows: DataFrame - provide the dataframe from the swing_highs_lows function
    close_break: bool - if True then the break of structure will be mitigated based on the close of the candle otherwise it will be the high/low.

    returns:
    BOS = 1 if bullish break of structure, -1 if bearish break of structure
    CHOCH = 1 if bullish change of character, -1 if bearish change of character
    Level = the level of the break of structure or change of character
    BrokenIndex = the index of the candle that broke the level
    """

    level_order = []
    highs_lows_order = []

    bos = np.zeros(len(ohlc), dtype=np.int32)
    choch = np.zeros(len(ohlc), dtype=np.int32)
    level = np.zeros(len(ohlc), dtype=np.float32)

    last_positions = []

    for i in range(len(swing_highs_lows["HighLow"])):
        if not np.isnan(swing_highs_lows["HighLow"][i]):
            level_order.append(swing_highs_lows["Level"][i])
            highs_lows_order.append(swing_highs_lows["HighLow"][i])
            if len(level_order) >= 4:
                # bullish bos
                bos[last_positions[-2]] = (
                    1
                    if (
                        np.all(highs_lows_order[-4:] == [-1, 1, -1, 1])
                        and np.all(
                            level_order[-4]
                            < level_order[-2]
                            < level_order[-3]
                            < level_order[-1]
                        )
                    )
                    else 0
                )
                level[last_positions[-2]] = (
                    level_order[-3] if bos[last_positions[-2]] != 0 else 0
                )

                # bearish bos
                bos[last_positions[-2]] = (
                    -1
                    if (
                        np.all(highs_lows_order[-4:] == [1, -1, 1, -1])
                        and np.all(
                            level_order[-4]
                            > level_order[-2]
                            > level_order[-3]
                            > level_order[-1]
                        )
                    )
                    else bos[last_positions[-2]]
                )
                level[last_positions[-2]] = (
                    level_order[-3] if bos[last_positions[-2]] != 0 else 0
                )

                # bullish choch
                choch[last_positions[-2]] = (
                    1
                    if (
                        np.all(highs_lows_order[-4:] == [-1, 1, -1, 1])
                        and np.all(
                            level_order[-1]
                            > level_order[-3]
                            > level_order[-4]
                            > level_order[-2]
                        )
                    )
                    else 0
                )
                level[last_positions[-2]] = (
                    level_order[-3]
                    if choch[last_positions[-2]] != 0
                    else level[last_positions[-2]]
                )

                # bearish choch
                choch[last_positions[-2]] = (
                    -1
                    if (
                        np.all(highs_lows_order[-4:] == [1, -1, 1, -1])
                        and np.all(
                            level_order[-1]
                            < level_order[-3]
                            < level_order[-4]
                            < level_order[-2]
                        )
                    )
                    else choch[last_positions[-2]]
                )
                level[last_positions[-2]] = (
                    level_order[-3]
                    if choch[last_positions[-2]] != 0
                    else level[last_positions[-2]]
                )

            last_positions.append(i)

    broken = np.zeros(len(ohlc), dtype=np.int32)
    for i in np.where(np.logical_or(bos != 0, choch != 0))[0]:
        mask = np.zeros(len(ohlc), dtype=np.bool_)
        # if the bos is 1 then check if the candles high has gone above the level
        if bos[i] == 1 or choch[i] == 1:
            mask = ohlc["close" if close_break else "high"][i + 2 :] > level[i]
        # if the bos is -1 then check if the candles low has gone below the level
        elif bos[i] == -1 or choch[i] == -1:
            mask = ohlc["close" if close_break else "low"][i + 2 :] < level[i]
        if np.any(mask):
            j = np.argmax(mask) + i + 2
            broken[i] = j

    # remove the ones that aren't broken
    for i in np.where(
        np.logical_and(np.logical_or(bos != 0, choch != 0), broken == 0)
    )[0]:
        bos[i] = 0
        choch[i] = 0
        level[i] = 0

    # replace all the 0s with np.nan
    bos = np.where(bos != 0, bos, np.nan)
    choch = np.where(choch != 0, choch, np.nan)
    level = np.where(level != 0, level, np.nan)
    broken = np.where(broken != 0, broken, np.nan)

    bos = pd.Series(bos, name="BOS")
    choch = pd.Series(choch, name="CHOCH")
    level = pd.Series(level, name="Level")
    broken = pd.Series(broken, name="BrokenIndex")

    return pd.concat([bos, choch, level, broken], axis=1)

@classmethod
def ob(
    cls,
    ohlc: DataFrame,
    swing_highs_lows: DataFrame,
    close_mitigation:bool = False,
) -> DataFrame:
    """
    OB - Order Blocks
    This method detects order blocks when there is a high amount of market orders exist on a price range.

    parameters:
    swing_highs_lows: DataFrame - provide the dataframe from the swing_highs_lows function
    close_mitigation: bool - if True then the order block will be mitigated based on the close of the candle otherwise it will be the high/low.

    returns:
    OB = 1 if bullish order block, -1 if bearish order block
    Top = top of the order block
    Bottom = bottom of the order block
    OBVolume = volume + 2 last volumes amounts
    Percentage = strength of order block (min(highVolume, lowVolume)/max(highVolume,lowVolume))
    """

    crossed = np.full(len(ohlc), False, dtype=bool)
    ob = np.zeros(len(ohlc), dtype=np.int32)
    top = np.zeros(len(ohlc), dtype=np.float32)
    bottom = np.zeros(len(ohlc), dtype=np.float32)
    obVolume = np.zeros(len(ohlc), dtype=np.float32)
    lowVolume = np.zeros(len(ohlc), dtype=np.float32)
    highVolume = np.zeros(len(ohlc), dtype=np.float32)
    percentage = np.zeros(len(ohlc), dtype=np.int32)
    mitigated_index = np.zeros(len(ohlc), dtype=np.int32)
    breaker = np.full(len(ohlc), False, dtype=bool)

    for i in range(len(ohlc)):
        close_index = i
        close_price = ohlc["close"].iloc[close_index]

        # Bullish Order Block
        if len(ob[ob == 1]) > 0:
            for j in range(len(ob) - 1, -1, -1):
                if ob[j] == 1:
                    currentOB = j
                    if breaker[currentOB]:
                        if ohlc.high.iloc[close_index] > top[currentOB]:
                            ob[j] = top[j] = bottom[j] = obVolume[j] = lowVolume[
                                j
                            ] = highVolume[j] = mitigated_index[j] = percentage[
                                j
                            ] = 0.0

                    elif (
                            not close_mitigation
                            and ohlc["low"].iloc[close_index] < bottom[currentOB]
                        ) or (
                            close_mitigation
                            and min(
                                ohlc["open"].iloc[close_index],
                                ohlc["close"].iloc[close_index],
                            )
                            < bottom[currentOB]
                        ):
                        breaker[currentOB] = True
                        mitigated_index[currentOB] = close_index - 1
        last_top_index = None
        for j in range(len(swing_highs_lows["HighLow"])):
            if swing_highs_lows["HighLow"][j] == 1 and j < close_index:
                last_top_index = j
        if last_top_index is not None:
            swing_top_price = ohlc["high"].iloc[last_top_index]
            if close_price > swing_top_price and not crossed[last_top_index]:
                crossed[last_top_index] = True
                obBtm = ohlc["high"].iloc[close_index - 1]
                obTop = ohlc["low"].iloc[close_index - 1]
                obIndex = close_index - 1
                for j in range(1, close_index - last_top_index):
                    obBtm = min(
                        ohlc["low"].iloc[last_top_index + j],
                        obBtm,
                    )
                    if obBtm == ohlc["low"].iloc[last_top_index + j]:
                        obTop = ohlc["high"].iloc[last_top_index + j]
                    obIndex = (
                        last_top_index + j
                        if obBtm == ohlc["low"].iloc[last_top_index + j]
                        else obIndex
                    )

                ob[obIndex] = 1
                top[obIndex] = obTop
                bottom[obIndex] = obBtm
                obVolume[obIndex] = (
                    ohlc["volume"].iloc[close_index]
                    + ohlc["volume"].iloc[close_index - 1]
                    + ohlc["volume"].iloc[close_index - 2]
                )
                lowVolume[obIndex] = ohlc["volume"].iloc[close_index - 2]
                highVolume[obIndex] = (
                    ohlc["volume"].iloc[close_index]
                    + ohlc["volume"].iloc[close_index - 1]
                )
                percentage[obIndex] = (
                    np.min([highVolume[obIndex], lowVolume[obIndex]], axis=0)
                    / np.max([highVolume[obIndex], lowVolume[obIndex]], axis=0)
                ) * 100.0

    for i in range(len(ohlc)):
        close_index = i
        close_price = ohlc["close"].iloc[close_index]

        # Bearish Order Block
        if len(ob[ob == -1]) > 0:
            for j in range(len(ob) - 1, -1, -1):
                if ob[j] == -1:
                    currentOB = j
                    if breaker[currentOB]:
                        if ohlc.low.iloc[close_index] < bottom[currentOB]:
                            ob[j] = top[j] = bottom[j] = obVolume[j] = lowVolume[
                                j
                            ] = highVolume[j] = mitigated_index[j] = percentage[
                                j
                            ] = 0.0

                    elif (
                            not close_mitigation
                            and ohlc["high"].iloc[close_index] > top[currentOB]
                        ) or (
                            close_mitigation
                            and max(
                                ohlc["open"].iloc[close_index],
                                ohlc["close"].iloc[close_index],
                            )
                            > top[currentOB]
                        ):
                        breaker[currentOB] = True
                        mitigated_index[currentOB] = close_index
        last_btm_index = None
        for j in range(len(swing_highs_lows["HighLow"])):
            if swing_highs_lows["HighLow"][j] == -1 and j < close_index:
                last_btm_index = j
        if last_btm_index is not None:
            swing_btm_price = ohlc["low"].iloc[last_btm_index]
            if close_price < swing_btm_price and not crossed[last_btm_index]:
                crossed[last_btm_index] = True
                obBtm = ohlc["low"].iloc[close_index - 1]
                obTop = ohlc["high"].iloc[close_index - 1]
                obIndex = close_index - 1
                for j in range(1, close_index - last_btm_index):
                    obTop = max(ohlc["high"].iloc[last_btm_index + j], obTop)
                    obBtm = (
                        ohlc["low"].iloc[last_btm_index + j]
                        if obTop == ohlc["high"].iloc[last_btm_index + j]
                        else obBtm
                    )
                    obIndex = (
                        last_btm_index + j
                        if obTop == ohlc["high"].iloc[last_btm_index + j]
                        else obIndex
                    )

                ob[obIndex] = -1
                top[obIndex] = obTop
                bottom[obIndex] = obBtm
                obVolume[obIndex] = (
                    ohlc["volume"].iloc[close_index]
                    + ohlc["volume"].iloc[close_index - 1]
                    + ohlc["volume"].iloc[close_index - 2]
                )
                lowVolume[obIndex] = (
                    ohlc["volume"].iloc[close_index]
                    + ohlc["volume"].iloc[close_index - 1]
                )
                highVolume[obIndex] = ohlc["volume"].iloc[close_index - 2]
                percentage[obIndex] = (
                    np.min([highVolume[obIndex], lowVolume[obIndex]], axis=0)
                    / np.max([highVolume[obIndex], lowVolume[obIndex]], axis=0)
                ) * 100.0

    ob = np.where(ob != 0, ob, np.nan)
    top = np.where(~np.isnan(ob), top, np.nan)
    bottom = np.where(~np.isnan(ob), bottom, np.nan)
    obVolume = np.where(~np.isnan(ob), obVolume, np.nan)
    mitigated_index = np.where(~np.isnan(ob), mitigated_index, np.nan)
    percentage = np.where(~np.isnan(ob), percentage, np.nan)

    ob_series = pd.Series(ob, name="OB")
    top_series = pd.Series(top, name="Top")
    bottom_series = pd.Series(bottom, name="Bottom")
    obVolume_series = pd.Series(obVolume, name="OBVolume")
    mitigated_index_series = pd.Series(mitigated_index, name="MitigatedIndex")
    percentage_series = pd.Series(percentage, name="Percentage")

    return pd.concat(
        [
            ob_series,
            top_series,
            bottom_series,
            obVolume_series,
            mitigated_index_series,
            percentage_series,
        ],
        axis=1,
    )

@classmethod
def liquidity(cls, ohlc: DataFrame, swing_highs_lows: DataFrame, range_percent:float = 0.01) -> Series:
    """
    Liquidity
    Liquidity is when there are multiply highs within a small range of each other.
    or multiply lows within a small range of each other.

    parameters:
    swing_highs_lows: DataFrame - provide the dataframe from the swing_highs_lows function
    range_percent: float - the percentage of the range to determine liquidity

    returns:
    Liquidity = 1 if bullish liquidity, -1 if bearish liquidity
    Level = the level of the liquidity
    End = the index of the last liquidity level
    Swept = the index of the candle that swept the liquidity
    """

    # subtract the highest high from the lowest low
    pip_range = (max(ohlc["high"]) - min(ohlc["low"])) * range_percent

    # go through all of the high level and if there are more than 1 within the pip range, then it is liquidity
    liquidity = np.zeros(len(ohlc), dtype=np.int32)
    liquidity_level = np.zeros(len(ohlc), dtype=np.float32)
    liquidity_end = np.zeros(len(ohlc), dtype=np.int32)
    liquidity_swept = np.zeros(len(ohlc), dtype=np.int32)

    for i in range(len(ohlc)):
        if swing_highs_lows["HighLow"][i] == 1:
            high_level = swing_highs_lows["Level"][i]
            range_low = high_level - pip_range
            range_high = high_level + pip_range
            temp_liquidity_level = [high_level]
            start = i
            end = i
            swept = 0
            for c in range(i + 1, len(ohlc)):
                if swing_highs_lows["HighLow"][c] == 1 and range_low <= swing_highs_lows["Level"][c] <= range_high:
                    end = c
                    temp_liquidity_level.append(swing_highs_lows["Level"][c])
                    swing_highs_lows.loc[c, "HighLow"] = 0
                if ohlc["high"].iloc[c] >= range_high:
                    swept = c
                    break
            if len(temp_liquidity_level) > 1:
                average_high = sum(temp_liquidity_level) / len(temp_liquidity_level)
                liquidity[i] = 1
                liquidity_level[i] = average_high
                liquidity_end[i] = end
                liquidity_swept[i] = swept

    # now do the same for the lows
    for i in range(len(ohlc)):
        if swing_highs_lows["HighLow"][i] == -1:
            low_level = swing_highs_lows["Level"][i]
            range_low = low_level - pip_range
            range_high = low_level + pip_range
            temp_liquidity_level = [low_level]
            start = i
            end = i
            swept = 0
            for c in range(i + 1, len(ohlc)):
                if swing_highs_lows["HighLow"][c] == -1 and range_low <= swing_highs_lows["Level"][c] <= range_high:
                    end = c
                    temp_liquidity_level.append(swing_highs_lows["Level"][c])
                    swing_highs_lows.loc[c, "HighLow"] = 0
                if ohlc["low"].iloc[c] <= range_low:
                    swept = c
                    break
            if len(temp_liquidity_level) > 1:
                average_low = sum(temp_liquidity_level) / len(temp_liquidity_level)
                liquidity[i] = -1
                liquidity_level[i] = average_low
                liquidity_end[i] = end
                liquidity_swept[i] = swept

    liquidity = np.where(liquidity != 0, liquidity, np.nan)
    liquidity_level = np.where(~np.isnan(liquidity), liquidity_level, np.nan)
    liquidity_end = np.where(~np.isnan(liquidity), liquidity_end, np.nan)
    liquidity_swept = np.where(~np.isnan(liquidity), liquidity_swept, np.nan)

    liquidity = pd.Series(liquidity, name="Liquidity")
    level = pd.Series(liquidity_level, name="Level")
    liquidity_end = pd.Series(liquidity_end, name="End")
    liquidity_swept = pd.Series(liquidity_swept, name="Swept")

    return pd.concat([liquidity, level, liquidity_end, liquidity_swept], axis=1)
    
@classmethod
def previous_high_low(cls, ohlc: DataFrame, time_frame: str = "1D") -> Series:
    """
    Previous High Low
    This method returns the previous high and low of the given time frame.

    parameters:
    time_frame: str - the time frame to get the previous high and low 15m, 1H, 4H, 1D, 1W, 1M

    returns:
    PreviousHigh = the previous high
    PreviousLow = the previous low
    """

    ohlc.index = pd.to_datetime(ohlc.index)

    resampled_ohlc = ohlc.resample(time_frame).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )

    # for every candle in ohlc add a new column with the previous high and low
    # Fix: Import the datetime module
    previous_high = np.zeros(len(ohlc), dtype=np.float32)
    previous_low = np.zeros(len(ohlc), dtype=np.float32)

    for i in range(len(ohlc)):
        current_time = ohlc.index[i]
        # get the 1st high where the current time is greater than the time from the resampled ohlc
        previous_high_index = resampled_ohlc["high"].where(
            resampled_ohlc.index < current_time
        ).last_valid_index()
        previous_high[i] = resampled_ohlc["high"][previous_high_index] if previous_high_index is not None else np.nan
        # get the 1st low where the current time is greater than the time from the resampled ohlc
        previous_low_index = resampled_ohlc["low"].where(
            resampled_ohlc.index < current_time
        ).last_valid_index()
        previous_low[i] = resampled_ohlc["low"][previous_low_index] if previous_low_index is not None else np.nan

    previous_high = pd.Series(previous_high, name="PreviousHigh")
    previous_low = pd.Series(previous_low, name="PreviousLow")

    return pd.concat([previous_high, previous_low], axis=1)

def analyze_smart_money(ohlc: pd.DataFrame) -> pd.DataFrame:
    """Обобщенная функция анализа смарт-мани."""
    ohlc['fvg'] = cls.fvg(ohlc)
    swing_highs_lows_result = cls.swing_highs_lows(ohlc)
    ohlc['bos_choch'] = cls.bos_choch(ohlc, swing_highs_lows_result)
    ohlc['ob'] = cls.ob(ohlc, swing_highs_lows_result)
    ohlc['liquidity'] = cls.liquidity(ohlc, swing_highs_lows_result)
    ohlc['previous_high_low'] = cls.previous_high_low(ohlc)

    return ohlc
