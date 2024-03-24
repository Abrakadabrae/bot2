import logging
import asyncio
from datetime import datetime
from aiogram import Bot, Dispatcher, types
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
import ccxt.async_support as ccxt
from functools import wraps
import numpy as np
import pandas as pd
from Smartmoney import smc
from pandas import DataFrame, Series
from datetime import datetime
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.utils import dropna
import requests
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import random
import os
import time


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_TOKEN = ''
API_KEY = ''
API_SECRET = ''
CRYPTOCOMPARE_API_KEY = ''

CRYPTOCOMPARE_URL = 'https://min-api.cryptocompare.com/data/price'

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, 10, 50)
toolbox.register("attr_float", random.uniform, 20, 50)
toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_int, toolbox.attr_float, toolbox.attr_float), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", lambda individual: (random.uniform(0, 1),))
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

population = toolbox.population(n=100)
NGEN = 40
for gen in range(NGEN):
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
    fits = map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))

best_ind = tools.selBest(population, 1)[0]
print("Best parameters:", best_ind)
print("Fitness:", best_ind.fitness.values)

async def perform_analysis(symbol):
    df = await fetch_historical_data(symbol)
    if df is not None and not df.empty:
        analyzed_df = analyze_smart_money(df)
        # Применение анализа "честных ценовых промежутков" (Fair Value Gap)
        df['fvg_info'] = df.apply(lambda x: smc.fvg(x), axis=1)
        # Применение анализа "высоких и низких качелей" (Swing Highs and Lows)
        df['swing_highs_lows'] = df.apply(lambda x: smc.swing_highs_lows(x, swing_length=50), axis=1)
        # Применение анализа "прорыва структуры и изменения характера" (BOS - Break of Structure, CHoCH - Change of Character)
        df['bos_choch_info'] = df.apply(lambda x: smc.bos_choch(x, df['swing_highs_lows'], close_break=True), axis=1)
        # Применение анализа "ордер блоков" (Order Blocks)
        df['ob_info'] = df.apply(lambda x: smc.ob(x, df['swing_highs_lows'], close_mitigation=False), axis=1)
        # Применение анализа "ликвидности" (Liquidity)
        df['liquidity_info'] = df.apply(lambda x: smc.liquidity(x, df['swing_highs_lows'], range_percent=0.01), axis=1)
        # Применение анализа "предыдущих максимумов и минимумов" (Previous High and Low)
        df['previous_high_low'] = df.apply(lambda x: smc.previous_high_low(x, time_frame="1D"), axis=1)
        
        # Пример объединения результатов анализа с исходным DataFrame
        analyzed_df = df.copy()
        analyzed_df['FVG'] = fvg_result['FVG']
        analyzed_df['SwingHL'] = swing_highs_lows_result['HighLow']

        # Добавляем результаты BOS и CHoCH
        analyzed_df['BOS'] = bos_choch_result['BOS']
        analyzed_df['CHOCH'] = bos_choch_result['CHOCH']
        analyzed_df['BOS_CHOCH_Level'] = bos_choch_result['Level']
        analyzed_df['BOS_CHOCH_BrokenIndex'] = bos_choch_result['BrokenIndex']

        # Добавляем результаты анализа Order Blocks
        analyzed_df['OB'] = ob_result['OB']
        analyzed_df['OB_Top'] = ob_result['Top']
        analyzed_df['OB_Bottom'] = ob_result['Bottom']
        analyzed_df['OB_Volume'] = ob_result['OBVolume']
        analyzed_df['OB_MitigatedIndex'] = ob_result['MitigatedIndex']
        analyzed_df['OB_Percentage'] = ob_result['Percentage']

        # Добавляем результаты анализа Liquidity
        analyzed_df['Liquidity'] = liquidity_result['Liquidity']
        analyzed_df['Liquidity_Level'] = liquidity_result['Level']
        analyzed_df['Liquidity_End'] = liquidity_result['End']
        analyzed_df['Liquidity_Swept'] = liquidity_result['Swept']

        # Добавляем результаты анализа предыдущих максимумов и минимумов
        analyzed_df['PreviousHigh'] = previous_high_low_result['PreviousHigh']
        analyzed_df['PreviousLow'] = previous_high_low_result['PreviousLow']
    else:
        print("Ошибка: DataFrame пустой или данные не получены.")
        
def fetch_price_cryptocompare(symbol):
    headers = {'Apikey': CRYPTOCOMPARE_API_KEY}
    params = {'fsym': symbol, 'tsyms': 'USD'}
    response = requests.get(CRYPTOCOMPARE_URL, headers=headers, params=params)
    data = response.json()
    price = data.get('USD')
    return price if price else None

async def fetch_historical_data(symbol, timeframe='1d', limit=100):
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'apiKey': API_KEY,
        'secret': API_SECRET,
    })
    since = exchange.parse8601('2021-01-01T00:00:00Z')
    candles = await exchange.fetch_ohlcv(symbol.upper() + '/USDT', timeframe, since, limit)
    await exchange.close()
    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

async def fetch_candles(symbol, timeframes):
    exchange = ccxt.binance({
        'apiKey': API_KEY,
        'secret': API_SECRET,
    })
    results = {}
    try:
        for timeframe in timeframes:
            candles = await exchange.fetch_ohlcv(symbol.upper() + '/USDT', timeframe, limit=100)
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            results[timeframe] = dropna(df)
    finally:
        await exchange.close()
    return results



def analyze_data(data):
    analysis_results = {}
    for timeframe, df in data.items():
        # Создаем экземпляры индикаторов и получаем значения
        macd_indicator = MACD(close=df['close'])
        df['momentum_macd'] = macd_indicator.macd()
        df['momentum_macd_signal'] = macd_indicator.macd_signal()
        df['momentum_macd_diff'] = macd_indicator.macd_diff()
        
        rsi_indicator = RSIIndicator(close=df['close'])
        df['momentum_rsi'] = rsi_indicator.rsi()
        
        bb = BollingerBands(close=df['close'])
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()

        # Новый анализ
        df['fvg_info'] = df.apply(lambda x: smc.fvg(x), axis=1)
        df['bos_choch_info'] = df.apply(lambda x: smc.bos_choch(x, df), axis=1)
        df['ob_info'] = df.apply(lambda x: smc.ob(x, df), axis=1)
        df['liquidity_info'] = df.apply(lambda x: smc.liquidity(x, df), axis=1)
        df['previous_high_low'] = df.apply(lambda x: smc.previous_high_low(x), axis=1)
        
        # Анализ ордер блоков
        df_with_order_blocks = find_order_blocks(df.copy()) # Используйте df.copy() для сохранения оригинального df неизменным
        
        analysis_results[timeframe] = {
            'original_df': df,  # Исходный DataFrame с добавленными индикаторами
            'df_with_order_blocks': df_with_order_blocks,  # DataFrame с ордер блоками
            'momentum_macd': df['momentum_macd'].iloc[-1],
            'momentum_macd_signal': df['momentum_macd_signal'].iloc[-1],
            'momentum_macd_diff': df['momentum_macd_diff'].iloc[-1],
            'momentum_rsi': df['momentum_rsi'].iloc[-1],
            'bb_high': df['bb_high'].iloc[-1],
            'bb_low': df['bb_low'].iloc[-1],
            # Сохраняем DataFrame только с ордер блоками для упрощения доступа
            'order_blocks': df_with_order_blocks[df_with_order_blocks['is_order_block']].copy()
        }
    return analysis_results

def generate_trade_signal(symbol, analysis_result, selected_timeframe, df_with_order_blocks):
    if selected_timeframe in analysis_result:
        df = analysis_result[selected_timeframe]['order_blocks']  # Получаем df с ордер блоками
        last_row = df.iloc[-1]
        
        direction = "🟩 LONG" if last_row['momentum_macd'] > 0 and last_row['momentum_rsi'] > 50 else "🟥 SHORT"
        entry_price = last_row['close']
        
        risk_ratio = 0.01  # Пример: риск 1%
        profit_ratio = risk_ratio * 3  # Прибыль 3%
    
        if direction == "🟩 LONG":
            target_price = entry_price * (1 + profit_ratio)
            stop_loss = entry_price * (1 - risk_ratio)
        else:
            target_price = entry_price * (1 - profit_ratio)
            stop_loss = entry_price * (1 + risk_ratio)
    
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        message = f"""
Анализ рынка на: {current_time}
Текущая цена для {symbol}/USDT: {entry_price:.2f}
Сигнал: {direction}
📈 Точка входа: {entry_price:.2f}
🎯 Цель: {target_price:.2f}
🚫 Стоп лосс: {stop_loss:.2f}
"""
    
        if not df.empty:
            message += "\nОбнаружены потенциальные ордер блоки на следующих уровнях цен:\n"
            for _, row in df.iterrows():
                block_type = "блок сопротивления" if row['block_type'] == 'resistance' else "блок поддержки"
                message += f"- Цена: {row['close']:.2f}, размер: {row['block_size']} ({block_type})\n"
        else:
            message += "\nОрдер блоки не обнаружены."
    else:
        message = "Данные для выбранного таймфрейма отсутствуют."

    return message.strip()

def get_analyse_keyboard():
    keyboard = InlineKeyboardMarkup()
    analyse_button = InlineKeyboardButton(text="Анализ", switch_inline_query_current_chat="/analyse ")
    keyboard.add(analyse_button)
    return keyboard

from aiogram.types import CallbackQuery

async def handle_analyze_command(message: types.Message):
    symbol = message.get_args().split()[0].upper()
    ohlc = await fetch_historical_data(symbol)  # Здесь ваша функция для получения данных
    if ohlc is not None:
        analyzed_ohlc = analyze_smart_money(ohlc)
        # Отправка результатов анализа пользователю
        await message.answer("Результаты анализа для " + symbol + "...")
    else:
        await message.answer("Ошибка: не удалось получить данные для " + symbol + ".")

@dp.callback_query_handler(lambda c: c.data.startswith('analyse:'))
async def handle_timeframe_selection(callback_query: types.CallbackQuery):
    await bot.answer_callback_query(callback_query.id)
    _, symbol, timeframe = callback_query.data.split(":")
    
    data = await fetch_candles(symbol, [timeframe])
    analysis_results = analyze_data(data)
    if timeframe in analysis_results:
        # Используем переменную `timeframe` непосредственно для получения результата анализа
        signal = generate_trade_signal(symbol, analysis_results, timeframe, analysis_results[timeframe]['order_blocks'])
        await bot.send_message(callback_query.from_user.id, signal)
    else:
        await bot.send_message(callback_query.from_user.id, f"Данные для {timeframe} недоступны.")

@dp.message_handler(commands=['analyse'])
async def prompt_timeframe_selection(message: types.Message):
    args = message.get_args().split()
    if not args:
        await message.reply("Введите символ для анализа. Например: /analyse BTCUSDT")
        return
    symbol = args[0].upper()
    
    # Создаем Inline-клавиатуру с выбором таймфреймов
    keyboard = InlineKeyboardMarkup(row_width=3)
    timeframes = ['15m', '30m', '1h', '4h', '1d']
    for timeframe in timeframes:
        callback_data = f"analyse:{symbol}:{timeframe}"
        keyboard.insert(InlineKeyboardButton(timeframe, callback_data=callback_data))
    
    await message.reply("Выберите таймфрейм для анализа:", reply_markup=keyboard)

@dp.message_handler(commands=['analyse'])
async def analyse_currency(message: types.Message):
    args = message.get_args() or ''
    args_list = args.split()
    if not args_list:
        await message.reply("Usage: /analyse <symbol>")
        return
    symbol = args_list[0].upper()

    # Получаем данные о ценах
    price_cryptocompare = fetch_price_cryptocompare(symbol)
  
    # Получаем данные для анализа
    timeframes = ['15m','30m','1h', '4h', '1d']
    data = await fetch_candles(symbol, timeframes)
    analysis_results = analyze_data(data)
    signal = generate_trade_signal(symbol, analysis_results['1h']['df'])  # Используем данные за 1 час для генерации сигнала
  
    # Формируем и отправляем сообщение
    price_info = f"CryptoCompare цена для {symbol}: ${price_cryptocompare}" if price_cryptocompare else f"{symbol} не найден на CryptoCompare."
    await message.reply(price_info + "\n" + signal)

async def main():
    await dp.start_polling()

if __name__ == "__main__":
    asyncio.run(main())
