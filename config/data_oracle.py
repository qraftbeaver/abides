import argparse
import numpy as np
import pandas as pd
import sys
import datetime as dt

from Kernel import Kernel
from util import util
from util.order import LimitOrder
from agent.ExchangeAgent import ExchangeAgent
from agent.examples.MarketReplayAgent import MarketReplayAgent
from agent.examples.MomentumAgent import MomentumAgent
from agent.NoiseAgent import NoiseAgent
from agent.ValueAgent import ValueAgent
from agent.examples.ImpactAgent import ImpactAgent
from util.oracle.DataOracle import DataOracle
from util.oracle.SparseMeanRevertingOracle import SparseMeanRevertingOracle
from agent.market_makers.AdaptiveMarketMakerAgent import AdaptiveMarketMakerAgent

########################################################################################################################
############################################### GENERAL CONFIG #########################################################

parser = argparse.ArgumentParser(description='Detailed options for market replay config.')

parser.add_argument('-c',
                    '--config',
                    required=True,
                    help='Name of config file to execute')
parser.add_argument('-t',
                    '--ticker',
                    required=True,
                    help='Name of the stock/symbol')
parser.add_argument('-d',
                    '--date',
                    required=True,
                    help='Historical date')
parser.add_argument('-l',
                    '--log_dir',
                    default=None,
                    help='Log directory name (default: unix timestamp at program start)')
parser.add_argument('-s',
                    '--seed',
                    type=int,
                    default=None,
                    help='numpy.random.seed() for simulation')
parser.add_argument('-v',
                    '--verbose',
                    action='store_true',
                    help='Maximum verbosity!')
parser.add_argument('--config_help',
                    action='store_true',
                    help='Print argument options for this config file')
# market maker
# market maker config
parser.add_argument('--mm-pov',
                    type=float,
                    default=0.025
                    )
parser.add_argument('--mm-window-size',
                    type=util.validate_window_size,
                    default='adaptive'
                    )
parser.add_argument('--mm-min-order-size',
                    type=int,
                    default=1
                    )
parser.add_argument('--mm-num-ticks',
                    type=int,
                    default=10
                    )
parser.add_argument('--mm-wake-up-freq',
                    type=str,
                    default='10S'
                    )
parser.add_argument('--mm-skew-beta',
                    type=float,
                    default=0
                    )
parser.add_argument('--mm-level-spacing',
                    type=float,
                    default=5
                    )
parser.add_argument('--mm-spread-alpha',
                    type=float,
                    default=0.75
                    )
parser.add_argument('--mm-backstop-quantity',
                    type=float,
                    default=50000)

parser.add_argument('--fund-vol',
                    type=float,
                    default=1e-8,
                    help='Volatility of fundamental time series.'
                    )




args, remaining_args = parser.parse_known_args()

if args.config_help:
    parser.print_help()
    sys.exit()

log_dir = args.log_dir  # Requested log directory.
seed = args.seed  # Random seed specification on the command line.
if not seed: seed = int(pd.Timestamp.now().timestamp() * 1000000) % (2 ** 32 - 1)
np.random.seed(seed)

util.silent_mode = not args.verbose
LimitOrder.silent_mode = not args.verbose

simulation_start_time = dt.datetime.now()
print("Simulation Start Time: {}".format(simulation_start_time))
print("Configuration seed: {}".format(seed))
print("Log Directory: {}".format(log_dir))
########################################################################################################################
############################################### AGENTS CONFIG ##########################################################

# Historical date to simulate.
historical_date = args.date
historical_date = pd.to_datetime(historical_date)
symbol = args.ticker
print("Symbol: {}".format(symbol))
print("Date: {}\n".format(historical_date))

agent_count, agents, agent_types = 0, [], []

# 1) Exchange Agent
mkt_open = historical_date + pd.to_timedelta('09:00:00')
mkt_close = historical_date+ pd.to_timedelta('17:00:00')

print("Market Open : {}".format(mkt_open))
print("Market Close: {}".format(mkt_close))

agents.extend([ExchangeAgent(id=0,
                             name="EXCHANGE_AGENT",
                             type="ExchangeAgent",
                             mkt_open=mkt_open,
                             mkt_close=mkt_close,
                             symbols=[symbol],
                             log_orders=True,
                             pipeline_delay=0,
                             computation_delay=0,
                             stream_history=10,
                             book_freq=0,
                             wide_book=1,
                             random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                       dtype='uint64')))])
agent_types.extend("ExchangeAgent")
agent_count += 1

# 2) Market Replay Agent

# file_name = f'DOW30/{symbol}/{symbol}.{historical_date}'
# orders_file_path = f'/efs/data/{file_name}'

# agents.extend([MarketReplayAgent(id=1,
#                                  name="MARKET_REPLAY_AGENT",
#                                  type='MarketReplayAgent',
#                                  symbol=symbol,
#                                  log_orders=False,
#                                  date=historical_date,
#                                  start_time=mkt_open,
#                                  end_time=mkt_close,
#                                  orders_file_path=orders_file_path,
#                                  processed_orders_folder_path='/efs/data/marketreplay/',
#                                  starting_cash=0,
#                                  random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
#                                                                                            dtype='uint64')))])
# agent_types.extend("MarketReplayAgent")
# agent_count += 1





# Momentum agents
num_momentum_agents = 6
starting_cash = 10000000
log_orders = None
agents.extend([MomentumAgent(id=j,
                             name="MOMENTUM_AGENT_{}".format(j),
                             type="MomentumAgent",
                             symbol=symbol,
                             starting_cash=starting_cash,
                             min_size=1,
                             max_size=10,
                             wake_up_freq='20s',
                             log_orders=log_orders,
                             random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                       dtype='uint64')))
               for j in range(agent_count, agent_count + num_momentum_agents)])
agent_count += num_momentum_agents
agent_types.extend("MomentumAgent")



# Oracle
# symbols = {symbol: {'r_bar': 5860000,
#                     'kappa': 1.67e-16,
#                     'sigma_s': 0,
#                     'fund_vol': 1e-8,
#                     'megashock_lambda_a': 2.77778e-18,
#                     'megashock_mean': 1e3,
#                     'megashock_var': 5e4,
#                     'random_state': np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64'))}}
symbols = {symbol: symbol}
oracle = DataOracle(mkt_open, mkt_close, symbols)
end_price = oracle.getDailyClosePrice(symbol, mkt_close, cents=True)
open_price = oracle.getDailyOpenPrice(symbol, mkt_open, cents=True)

# oracle = None

# 3) Value Agents
num_value = 10

r_bar = end_price
sigma_n = r_bar / 10
kappa = 1.67e-15
lambda_a = 7e-11
agents.extend([ValueAgent(id=j,
                          name="Value Agent {}".format(j),
                          type="ValueAgent",
                          symbol=symbol,
                          starting_cash=starting_cash,
                          sigma_n=sigma_n,
                          r_bar=r_bar,
                          kappa=1.67e-16,
                          lambda_a=lambda_a,
                          log_orders=log_orders,
                          random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')))
               for j in range(agent_count, agent_count + num_value)])
agent_count += num_value
agent_types.extend(['ValueAgent'])



# 4) Market Maker Agents

"""
window_size ==  Spread of market maker (in ticks) around the mid price
pov == Percentage of transacted volume seen in previous `mm_wake_up_freq` that
       the market maker places at each level
num_ticks == Number of levels to place orders in around the spread
wake_up_freq == How often the market maker wakes up

"""

# each elem of mm_params is tuple (window_size, pov, num_ticks, wake_up_freq, min_order_size)
mm_params = [(args.mm_window_size, args.mm_pov, args.mm_num_ticks, args.mm_wake_up_freq, args.mm_min_order_size),
             (args.mm_window_size, args.mm_pov, args.mm_num_ticks, args.mm_wake_up_freq, args.mm_min_order_size)
             ]

num_mm_agents = len(mm_params)
mm_cancel_limit_delay = 50  # 50 nanoseconds

agents.extend([AdaptiveMarketMakerAgent(id=j,
                                name="ADAPTIVE_POV_MARKET_MAKER_AGENT_{}".format(j),
                                type='AdaptivePOVMarketMakerAgent',
                                symbol=symbol,
                                starting_cash=starting_cash,
                                pov=mm_params[idx][1],
                                min_order_size=mm_params[idx][4],
                                window_size=mm_params[idx][0],
                                num_ticks=mm_params[idx][2],
                                wake_up_freq=mm_params[idx][3],
                                cancel_limit_delay=mm_cancel_limit_delay,
                                skew_beta=args.mm_skew_beta,
                                level_spacing=args.mm_level_spacing,
                                spread_alpha=args.mm_spread_alpha,
                                backstop_quantity=args.mm_backstop_quantity,
                                log_orders=log_orders,
                                random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                          dtype='uint64')))
               for idx, j in enumerate(range(agent_count, agent_count + num_mm_agents))])
agent_count += num_mm_agents
agent_types.extend('POVMarketMakerAgent')


# Noise agents
num_noise = 500
# noise_mkt_open = historical_date + pd.to_timedelta("09:00:00")  # These times needed for distribution of arrival times
                                                                # of Noise Agents
# noise_mkt_close = historical_date + pd.to_timedelta("16:00:00")

# import pdb
# pdb.set_trace()
agents.extend([NoiseAgent(id=j,
                          name="NoiseAgent {}".format(j),
                          type="NoiseAgent",
                          symbol=symbol,
                          starting_cash=starting_cash,
                          wakeup_time=util.get_wake_time(mkt_open, mkt_close),
                          log_orders=log_orders,
                          random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')))
               for j in range(agent_count, agent_count + num_noise)])
agent_count += num_noise
agent_types.extend(['NoiseAgent'])

'''
# impact agent
i = agent_count
impact_time = noise_mkt_open = historical_date + pd.to_timedelta("10:00:00") 
agents.append(ImpactAgent(i, "Impact Agent {}".format(i), "ImpactAgent", symbol = symbol, starting_cash = starting_cash, greed = 0.5, impact = True, impact_time = impact_time, random_state = np.random.RandomState(seed=np.random.randint(low=0,high=2**32))))
agent_types.append("Impact Agent {}".format(i))
agent_count += 1
'''

########################################################################################################################
########################################### KERNEL AND OTHER CONFIG ####################################################

kernel = Kernel("Market Replay Kernel", random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                                  dtype='uint64')))

kernelStartTime = historical_date
kernelStopTime = historical_date + pd.to_timedelta('13:00:00')

defaultComputationDelay = 0
latency = np.zeros((agent_count, agent_count))
noise = [0.0]

kernel.runner(agents=agents,
              startTime=kernelStartTime,
              stopTime=kernelStopTime,
              agentLatency=latency,
              latencyNoise=noise,
              defaultComputationDelay=defaultComputationDelay,
              defaultLatency=0,
              oracle=oracle,
              log_dir=args.log_dir)

simulation_end_time = dt.datetime.now()
print("Simulation End Time: {}".format(simulation_end_time))
print("Time taken to run simulation: {}".format(simulation_end_time - simulation_start_time))
