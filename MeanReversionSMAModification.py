# Import the libraries .
import quantopian.algorithm as algo
import quantopian.optimize as opt
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import Returns, SimpleMovingAverage
from quantopian.pipeline.filters import QTradableStocksUS
from quantopian.pipeline.data.psychsignal import stocktwits

import numpy as np

# Define static variables that can be accessed in the rest of the algorithm.

# Controls the maximum leverage of the algorithm. A value of 1.0 means the algorithm should spend no more than its starting capital (doesn't borrow money).
MAX_GROSS_EXPOSURE = 1.0

# Controls the maximum percentage of the portfolio that can be invested in any one security. 
# A value of 0.025 means the portfolio will invest a maximum of 2.5% of its portfolio in any one stock.
MAX_POSITION_CONCENTRATION = 0.025

# Controls the lookback window length of the Returns factor used by this algorithm to rank stocks.
RETURNS_LOOKBACK_DAYS = 7


def initialize(context):
    # Rebalance on the first trading day of each week at 11AM.
    algo.schedule_function(
        rebalance,
        algo.date_rules.week_start(days_offset=0),
        algo.time_rules.market_open(hours=1, minutes=30)
    )

    # Create and attach our pipeline (dynamic stock selector), defined below.
    algo.attach_pipeline(make_pipeline(context), 'mean_reversion_example')


def make_pipeline(context):

    universe = QTradableStocksUS()
    
    # Mean Average Multiplier calculation
    mean_close_50 = SimpleMovingAverage(inputs=[USEquityPricing.close], window_length=50, mask=universe)
    mean_close_200 = SimpleMovingAverage(inputs=[USEquityPricing.close], window_length=200, mask=universe)
    
    mean_average_multiplier = mean_close_200/mean_close_50  
    
    #Sentiment Score calculation
    bull_m_bear = (SimpleMovingAverage(inputs=[stocktwits.bull_minus_bear], window_length= RETURNS_LOOKBACK_DAYS, mask=universe)) 
    total_messages = (SimpleMovingAverage(inputs=[stocktwits.total_scanned_messages], window_length= RETURNS_LOOKBACK_DAYS, mask=universe))
                                  
    sentiment_score = total_messages / (bull_m_bear)
    sentiment_score_winsorized = sentiment_score.winsorize(min_percentile=0.01, max_percentile=0.95, mask=universe)
    
    
    
    #Returns with lookback calculations
    recent_returns = Returns(
        window_length= RETURNS_LOOKBACK_DAYS,
        mask = universe
    )
    recent_returns_zscore = recent_returns.zscore()
    

    #Universe and low-high calculations
    low_returns = recent_returns_zscore.percentile_between(0,25, mask=universe)
    high_returns = recent_returns_zscore.percentile_between(75,100, mask=universe)

    securities_to_trade = (low_returns | high_returns) & universe
    
    #Total score calculation
    total_score = sentiment_score_winsorized*recent_returns_zscore*mean_average_multiplier

    pipe = Pipeline(
        columns={
            'total_score': total_score,
        },
        screen = securities_to_trade
    )

    return pipe

def before_trading_start(context, data):
    # pipeline_output returns a pandas DataFrame with the results of our factors and filters.
    context.output = algo.pipeline_output('mean_reversion_example').fillna(1)

    # Sets the list of securities we want to long as the securities with a 'True' value in the low_returns column.
    context.total_score = context.output['total_score'].replace([np.inf,-np.inf], 1)


def rebalance(context, data):

    # Each day, we will enter and exit positions by defining a portfolio optimization problem. 
    # To do that, we need to set an objective for our portfolio as well as a series of constraints. 
    
    # Our objective is to maximize alpha, where 'alpha' is defined by the negative of total_score factor.
    objective = opt.MaximizeAlpha(-context.total_score)
    
    # We want to constrain our portfolio to invest a maximum total amount of money (defined by MAX_GROSS_EXPOSURE).
    max_gross_exposure = opt.MaxGrossExposure(MAX_GROSS_EXPOSURE)
    
    # We want to constrain our portfolio to invest a limited amount in any one position. 
    #To do this, we constrain the position to be between +/- 
    # MAX_POSITION_CONCENTRATION (on Quantopian, a negative weight corresponds to a short position).
    max_position_concentration = opt.PositionConcentration.with_equal_bounds(
        -MAX_POSITION_CONCENTRATION,
        MAX_POSITION_CONCENTRATION
    )
    
    # We want to constraint our portfolio to be dollar neutral (equal amount invested in long and short positions).
    dollar_neutral = opt.DollarNeutral()
    
    # Stores all of our constraints in a list.
    constraints = [
        max_gross_exposure,
        max_position_concentration,
        dollar_neutral,
    ]

    algo.order_optimal_portfolio(objective, constraints)
