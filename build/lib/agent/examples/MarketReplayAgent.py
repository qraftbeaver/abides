import pickle
import os.path
from datetime import datetime, timedelta
import pandas as pd
#from joblib import Memory

from agent.TradingAgent import TradingAgent
from util.order.LimitOrder import LimitOrder
from util.util import log_print


class MarketReplayAgent(TradingAgent):

    def __init__(self, id, name, type, symbol, date, start_time, end_time,
                 orders_file_path, processed_orders_folder_path,
                 starting_cash, log_orders=False, random_state=None):
        super().__init__(id, name, type, starting_cash=starting_cash, log_orders=log_orders, random_state=random_state)
        self.symbol = symbol
        self.date = date
        self.log_orders = log_orders
        self.executed_trades = dict()
        self.state = 'AWAITING_WAKEUP'

        self.x = L3OrdersProcessor(self.symbol,
                                                   self.date, start_time, end_time,
                                                   orders_file_path, processed_orders_folder_path)
        self.wakeup_times = self.x.wakeup_times

    def wakeup(self, currentTime):
        super().wakeup(currentTime)
        if not self.mkt_open or not self.mkt_close:
            return
        try:
            self.setWakeup(self.wakeup_times[0])
            self.wakeup_times.pop(0)
            self.placeOrder(currentTime, self.x.orders_dict[currentTime])
        except IndexError:
            log_print(f"Market Replay Agent submitted all orders - last order @ {currentTime}")

    def receiveMessage(self, currentTime, msg):
        super().receiveMessage(currentTime, msg)
        if msg.body['msg'] == 'ORDER_EXECUTED':
            order = msg.body['order']
            self.executed_trades[currentTime] = [order.fill_price, order.quantity]
            self.last_trade[self.symbol] = order.fill_price

    def placeOrder(self, currentTime, order):
        if len(order) == 1:
            order = order[0]
            order_id = order['ORDER_ID']
            existing_order = self.orders.get(order_id)
            # import pdb
            # pdb.set_trace()
            if not existing_order and order['SIZE'] > 0 and order['TYPE']==1:
                self.placeLimitOrder(self.symbol, order['SIZE'], order['BUY_SELL_FLAG'] == 'BUY', order['PRICE'],
                                     order_id=order_id)
            # elif existing_order and order['SIZE'] == 0:
            elif existing_order and order['TYPE'] == 3:
                self.cancelOrder(existing_order)
            elif existing_order and order['TYPE'] == 2 :
                self.modifyOrder(existing_order, LimitOrder(self.id, currentTime, self.symbol, order['SIZE'],
                                                            order['BUY_SELL_FLAG'] == 'BUY', order['PRICE'],
                                                            order_id=order_id))
        else:
            for ind_order in order:
                self.placeOrder(currentTime, order=[ind_order])

    def getWakeFrequency(self):
        log_print(f"Market Replay Agent first wake up: {self.x.first_wakeup}")
        return self.x.first_wakeup - self.mkt_open


#mem = Memory(cachedir='./cache', verbose=0)


class L3OrdersProcessor:
    COLUMNS = ['TIMESTAMP', 'ORDER_ID', 'PRICE', 'SIZE', 'BUY_SELL_FLAG']
    DIRECTION = {1: 'BUY', -1: 'SELL'}

    # Class for reading historical exchange orders stream
    def __init__(self, symbol, date, start_time, end_time, orders_file_path, processed_orders_folder_path):
        self.symbol = symbol
        self.date = date
        self.start_time = start_time
        self.end_time = end_time
        # self.orders_file_path = orders_file_path
        self.orders_file_path =  '/home/qraftbeaver/Research/abides/data/lobster/AAPL_order.csv'
        # self.processed_orders_folder_path = processed_orders_folder_path
        self.processed_orders_folder_path = '/home/qraftbeaver/Research/abides/data/lobster/AAPL_order.pkl'

        self.orders_dict = self.processOrders()
        self.wakeup_times = [*self.orders_dict]
        self.first_wakeup = self.wakeup_times[0]

    def processOrders(self):
        def convertDate(date_str):
            try:
                return datetime.fromtimestamp(date_str) + timedelta( days=17885, hours=14, minutes=30)
            except ValueError:
                return convertDate(date_str[:-1])

        #@mem.cache
        def read_processed_orders_file(processed_orders_file):
            with open(processed_orders_file, 'rb') as handle:
                return pickle.load(handle)

        # processed_orders_file = f'{self.processed_orders_folder_path}marketreplay_{self.symbol}_{self.date.date()}.pkl'
        processed_orders_file = f'{self.processed_orders_folder_path}'
        if os.path.isfile(processed_orders_file):
            print(f'Processed file exists for {self.symbol} and {self.date.date()}: {processed_orders_file}')
            return read_processed_orders_file(processed_orders_file)
        else:
            print(f'Processed file does not exist for {self.symbol} and {self.date.date()}, processing...')
            # orders_df = pd.read_csv(self.orders_file_path).iloc[1:]
            # all_columns = orders_df.columns[0].split('|')
            # orders_df = orders_df[orders_df.columns[0]].str.split('|', 16, expand=True)
            # orders_df.columns = all_columns
            # orders_df = orders_df[L3OrdersProcessor.COLUMNS]
            orders_df = pd.read_csv(self.orders_file_path)
            orders_df['BUY_SELL_FLAG'] = orders_df['BUY_SELL_FLAG'].astype(int).replace(L3OrdersProcessor.DIRECTION)
            orders_df['TIMESTAMP'] = orders_df['TIMESTAMP'].astype(float).apply(convertDate)
            orders_df['SIZE'] = orders_df['SIZE'].astype(int)
            # orders_df['PRICE'] = orders_df['PRICE'].astype(float) * 100
            orders_df['PRICE'] = orders_df['PRICE'].astype(int)
            orders_df['TYPE'] = orders_df['TYPE'].astype(int)
            '''
            2.) Type:
				1: Submission of a new limit order
				2: Cancellation (Partial deletion 
				   of a limit order)
				3: Deletion (Total deletion of a limit order)
				4: Execution of a visible limit order			   	 
				5: Execution of a hidden limit order
				7: Trading halt indicator 				   
				   (Detailed information below)
            '''
            orders_df['ORDER_ID'] = orders_df['ID'].astype(int)

            orders_df = orders_df.loc[(orders_df.TIMESTAMP >= self.start_time) & (orders_df.TIMESTAMP < self.end_time)]
            orders_df.set_index('TIMESTAMP', inplace=True)
            log_print(f"Number of Orders: {len(orders_df)}")
            orders_dict = {k: g.to_dict(orient='records') for k, g in orders_df.groupby(level=0)}
            with open(processed_orders_file, 'wb') as handle:
                pickle.dump(orders_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print(f'processed file created as {processed_orders_file}')
            return orders_dict
