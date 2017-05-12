import sys, getopt
import random
import pandas as pd
import data_utilities as du
import visuals as vs
from datetime import datetime, timedelta

class Portfolio(object):
    def __init__(self, investment=10000.00, stocks=['AAPL']):
        self.initial_investment = investment
        self.stocks = stocks
        self.holdings = pd.DataFrame(index=stocks,columns=['cash','shares','initial_price','buy_price','price'])
        self.holdings['cash'] = self.initial_investment
        self.holdings['shares'] = 0.0
        self.holdings['price'] = 0.0
        self.holdings['buy_price'] = 0.0
        self.holdings['initial_price'] = 0.0
        self.transaction_history = pd.DataFrame(columns=['Date','Symbol','Action','Shares','Price','Gain/Loss'])
        self.tindex = 0

    def reset(self):
        self.holdings['cash'] = self.initial_investment
        self.holdings['shares'] = 0.0
        self.holdings['price'] = 0.0
        self.holdings['buy_price'] = 0.0
        self.holdings['initial_price'] = 0.0

    def set_price(self,stock, price):
        self.holdings.loc[stock]['price'] = price
        if self.holdings.loc[stock]['initial_price'] == 0.0:
            self.holdings.loc[stock]['initial_price'] = price
            

    def has_shares(self,stock):
        return (self.holdings.loc[stock]['shares'] > 0.0)
        

    def sell_stock(self,stock,date):

        p = self.holdings.loc[stock]['price']
        s = self.holdings.loc[stock]['shares']
        bp = self.holdings.loc[stock]['buy_price']
        self.holdings.loc[stock]['cash'] = p*s
        self.holdings.loc[stock]['shares'] = 0.0
        self.transaction_history.loc[self.tindex] = [date, stock,'sell',s, p, (p-bp)*s]
        self.tindex += 1

    def buy_stock(self,stock,date):
        c = self.holdings.loc[stock]['cash']
        p = self.holdings.loc[stock]['price']
        self.holdings.loc[stock]['shares'] = c/p
        self.holdings.loc[stock]['cash'] = 0.0
        self.holdings.loc[stock]['buy_price'] = p
        self.transaction_history.loc[self.tindex] = [date, stock,'buy',c/p,p, 0.0]
        self.tindex += 1

        
        
    def value(self):
       
        return sum(self.holdings['price']*self.holdings['shares']+self.holdings['cash'])
        
    def buy_hold_value(self):
        return sum(self.holdings['price']/self.holdings['initial_price']*self.initial_investment)
       
        
        
        
    
class LearningTrader(object):
        # A Trader that learns to trade in the market
        
    def __init__(self,  learning=False, epsilon=1.0, alpha=0.5, investment=10000.00, stocks='AAPL', verbose=False,features = ['MA(20)']):
#        super(LearningTrader, self).__init__(env)     # Set the Trader in the evironment 
#        self.planner = RoutePlanner(self.env, self)  # Create a route planner
        self.portfolio = Portfolio(investment,stocks)
        self.valid_actions = [None,'buy','sell']  # The set of valid actions

        # Set parameters of the learning Trader
        self.learning = learning # Whether the Trader is expected to learn
        self.Q = dict()          # Create a Q-table which will be a dictionary of tuples
        self.epsilon = epsilon   # Random exploration factor
        self.alpha = alpha       # Learning factor
        self.verbose = verbose
        self.log_data = pd.DataFrame({'q_states':[0],'avg_reward':[0.0],'epsilon':[0.0]})
        #self.features = ['MA(20)','UB(20)','LB(20)','MA(50)','UB(50)','LB(50)']
        #self.features = ['MA(20)']
        self.features = features
        

        ###########
        ## TO DO ##
        ###########
        # Set any additional class parameters as needed
        self.decay = 0.05
        self.trial = 0


    def reset(self, testing=False):
        """ The reset function is called at the beginning of each trial.
            'testing' is set to True if testing trials are being used
            once training trials have completed. """

        
        
        ########### 
        ## TO DO ##
        ###########
        # Update epsilon using a decay function of your choice
        # Update additional class parameters as needed
        # If 'testing' is True, set epsilon and alpha to 0
        if testing:
            self.epsilon = 0.0
            self.alpha = 0.0
            # Reset the portfolio
            self.portfolio.reset()
    	else:
             self.trial += 1
             # initial learning trial
#             self.epsilon -= self.decay
             #optimized trail
             self.epsilon = pow(0.998, self.trial)
             


        return None

    def build_state(self,stock, inputs):
        # these are the data points we will use for the state
        # they will be compared to the ajusted close price
        if self.verbose:
            print "Building state for {} with inputs ".format(stock)
#            print inputs['Adj_Close'], inputs['MA(20)']
            print inputs['Adj_Close'], inputs[self.features]
        # Collect data about the environment
        has_stock = self.portfolio.has_shares(stock) # Shares held for the stock
        
        # compute adjusted close relative to selected features
        statel = [has_stock]
        for feature in self.features:
            if inputs['Adj_Close'] > inputs[feature]:
                statel.append('above')
            elif inputs['Adj_Close'] < inputs[feature]:
                statel.append('below')
            else:
                statel.append('equal')  #might be able to just roll this into the > case to reduce states
           
        # compute adjusted close relative to moving average 50

        # Set 'state' as a tuple of relevant data for the Trader
        state = tuple(statel)
        
        if self.verbose:
            print "Current state is {}".format(state)
 
        return state


    def get_maxQ(self, state):
        """ The get_max_Q function is called when the Trader is asked to find the
            maximum Q-value of all actions based on the 'state' the smartcab is in. """

        ########### 
        ## TO DO ##
        ###########
        # Calculate the maximum Q-value of all actions for a given state

        maxQ = max(self.Q[state].values())
        
        if self.verbose:
            print "MaxQ for state {} is {}.".format(state,maxQ)

        return maxQ 


    def createQ(self, state):
        """ The createQ function is called when a state is generated by the Trader. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, check if the 'state' is not in the Q-table
        # If it is not, create a new dictionary for that state
        #   Then, for each action available, set the initial Q-value to 0.0
        #print state
        if self.learning:
        	if not state in self.Q:
        		self.Q[state] = dict()
        		for action in self.valid_actions:
        			self.Q[state][action] = 0.0

	return


    def choose_action(self, state):
        """ The choose_action function is called when the Trader is asked to choose
            which action to take, based on the 'state' the smartcab is in. """

        # Set the Trader state and default action
        self.state = state
#        self.next_waypoint = self.planner.next_waypoint()
        action = None

        ########### 
        ## TO DO ##
        ###########
        # When not learning, choose a random action
        # When learning, choose a random action with 'epsilon' probability
        #   Otherwise, choose an action with the highest Q-value for the current state
        if self.verbose:
            print "Choosing action with learning {}, state {} and epsilon {}".format(self.learning, self.state, self.epsilon)
        if not self.learning:
            action = random.choice(self.valid_actions)
        elif random.random() < self.epsilon:
            action = random.choice(self.valid_actions)
        else:
            
            maxQ = self.get_maxQ(state)
            
            # Find the actions that match maxQ, if multiple then chose random one
            # from the set
            actionset = []
            
            for name in self.Q[state]:
                if self.Q[state][name] == maxQ:
                    actionset.append(name)
                    
            action = random.choice(actionset)
            
        
        if self.verbose:
           print "Action chosen ", action
           
	return action
 
    def explain_policy(self):
        
        print "Policy ----"
        for state in self.Q:
            maxQ = self.get_maxQ(state)
            
            # Find the actions that match maxQ
            # from the set
            actionset = []
            
            for name in self.Q[state]:
                if self.Q[state][name] == maxQ:
                    actionset.append(name)
            
            policy = "If has_stock is {} ".format(state[0])
            for idx, feature in enumerate(self.features):
                policy = policy + " and price is {} {}".format(state[idx+1],feature)
            policy = policy + " then {}. reward({})".format(actionset,maxQ)
            #print "If has_stock is {} and price is {} moving average(20) then {}".format(state[0],state[1],actionset)
            print policy
         
    def learn(self, state, action, reward):
        """ The learn function is called after the Trader completes an action and
            receives an award. This function does not consider future rewards 
            when conducting learning. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, implement the value iteration update rule
        #   Use only the learning rate 'alpha' (do not use the discount factor 'gamma')
        if self.learning:
            
            self.Q[state][action] = (1 - self.alpha ) * self.Q[state][action] + self.alpha * reward 
            if self.verbose:
                print "Q {} for state {} action {}".format(self.Q[state][action],state,action)
    
        return

    def act(self, stock, date, action, nextday):
       
        
        #nextday = np.sign(nextday)  # quantize this. originally used full value
        #reward = nextday # use the daily return as reward, this is the default for a 'buy'
        has_stock = self.portfolio.has_shares(stock)
        if action == 'buy':
            if has_stock:               # we already have it, big penalyt for trying to buy it
                reward = -10.0
            else:
                reward = nextday        # reward if it goes up next day, otherwise penalize
                self.portfolio.buy_stock(stock,date)
        elif action == 'sell': 
            if not has_stock:           # we don't have it so big penalyt for trying to sell it 
                reward = -10.0
            else:
                reward = -nextday       # reward if it goes down next day, otherwise penalize
                self.portfolio.sell_stock(stock,date)    
        else:  #action is None
            if has_stock:
                reward = nextday        # penalize if we held and it goes down, otherwise reward
            else:
                reward = -nextday       # reqard if we didn't buy and it goes down, else penalize
        if self.verbose:
            print "Computed {} reward for stock {}, has-stock {}, date {}, action {} and next day returns {}".format(reward,stock,has_stock,date,action,nextday)
        return reward
                
        

    def update(self, stock, date, inputs, nextday):
        """ The update function is called when a time step is completed in the 
            environment for a given trial. This function will build the Trader
            state, choose an action, receive a reward, and learn if enabled. """

        state = self.build_state(stock,inputs)          # Get current state
        self.portfolio.set_price(stock,inputs['Adj_Close'])
        self.createQ(state)                 # Create 'state' in Q-table
        action = self.choose_action(state)  # Choose an action
        reward = self.act(stock, date, action, nextday) # Receive a reward
        self.learn(state, action, reward)   # Q-learn
        q_states = sum(len(v) for v in self.Q.itervalues())
        total_reward = sum(sum(v.values()) for v in self.Q.itervalues())
        self.log_data = self.log_data.append({'q_states':q_states,'avg_reward':total_reward/float(q_states),'epsilon':self.epsilon},ignore_index=True)
        self.portfolio.set_price(stock,inputs['Adj_Close'])
        return

def simulate(trader,stocks,stockdata, start_date, end_date, mode = 'training',verbose=False):
    
    if mode == 'testing':
      trader.reset(testing=True)  
    

    initial_value = trader.portfolio.value()
    print "Starting {} with portfolio value {:,.2f}".format(mode,initial_value)
    print "Holdings before {}------".format(mode)
    print trader.portfolio.holdings
  
    # run the simulation
    #split = len(stockdata[stocks[0]]['Adj_Close']) * 8 / 10
    for symbol, df in stockdata.iteritems():
        traindf = df[start_date : end_date]
        print "{} on stock {} for {} days from {} to {}".format(mode, symbol,len(traindf),start_date, end_date)
        current_return = None
        priorinputs = pd.DataFrame()
        for date, inputs in traindf.iterrows():
            if current_return is not None:
                #current_return = inputs['Daily_Return']
                current_return = (inputs['MA(20)_Return'] if mode == 'training' else 0.0)
                #current_return = inputs['MA(20)_Return']
                trader.update(symbol,date, priorinputs, current_return)
                if mode == 'training':
                    trader.reset()
            else: # skip the first record
                #current_return = inputs['Daily_Return']
                current_return = inputs['MA(20)_Return']
            priorinputs = inputs
            
        th = trader.portfolio.transaction_history[trader.portfolio.transaction_history['Symbol']==symbol]
        if verbose:
            print th
        title = '{} - Stock Prices for {}'.format(mode,symbol)
        plt_items = ['Adj_Close']
        plt_items.extend(trader.features) #['Adj_Close','MA(20)']
        vs.plot_data(traindf[plt_items],title=title, normalize=False,actions=th) 
        
    ending_value = trader.portfolio.value()
    print "Ending {} with portfolio value {:,.2f}".format(mode,ending_value)
    print "Net return from {} {:,.2f}".format(mode,ending_value-initial_value)
    print "Compared to buy/hold return of {:,.2f}.".format(trader.portfolio.buy_hold_value()-initial_value)
    print "Holdings after {}------".format(mode)
    print trader.portfolio.holdings
    print "Policies after {}------".format(mode)
    if verbose:
        print trader.Q
    print trader.explain_policy()
    
   

def run(stocks=['AAPL'],initial_investment=10000.00,
        start_train='2015-01-01',end_train='2015-12-31',
        start_test='2016-01-01',end_test='2016-12-31',verbose=False,features=['MA(20)']):
    
    # set parameters - will add command line choices later
    # could add validation for date ranges as well
    #start_date='2014-01-01' # note first 200 days get chopped off to compute moving average
    #end_date='2016-12-31'
    #stocks = ['AAPL','GE','IBM','GOOGL','XOM','GS']
    #initial_investment = 10000.00
    
    start_load = str((datetime.strptime(start_train, '%Y-%m-%d')-timedelta(days=300)).date())
    end_load = end_test
    
    #get the data
    print "Retrieving stock data for stocks {} from {} to {}".format(stocks,start_load,end_load)
    historydf = du.get_data(stocks,start_load,end_load)
    vs.plot_data(historydf,normalize=True)
    del historydf['SPY'] # get data adds this for comparison - should add logic to leave it if actually desired
    stocks.remove('SPY')
    
    if verbose:
        print historydf.describe()
    #dates = pd.date_range(start_date, end_date)
    #calculate averages
    print "Computing averages and other data points"
    stockdata = du.compute_features(historydf)
    if verbose:
        for symbol, df  in stockdata.iteritems():
            print "Financial data for ", symbol
            print df.describe()
            print df[:100]
    
    
    ##############
    # Create the artificial Trader
    # Flags:
    #   learning   - set to True to force the driving Trader to use Q-learning
    #    * epsilon - continuous value for the exploration factor, default is 1
    #    * alpha   - continuous value for the learning rate, default is 0.5
    #Trader = env.create_Trader(LearningTrader,learning = True, epsilon = 1, alpha = 0.5)
    Trader = LearningTrader(learning = True, epsilon = 1, alpha = 0.25, 
                               investment=initial_investment, stocks=stocks, verbose=verbose, features=features)
    
    simulate(Trader,stocks,stockdata,start_train,end_train,mode='training')
    
        
    #test the trader
    simulate(Trader,stocks,stockdata,start_test,end_test,mode='testing',verbose=verbose)
    
    vs.plot_log(Trader.log_data)

    
def main(argv):
    
    
    stocks=['AAPL']
    initial_investment=10000.00
    start_train='2015-01-01'
    end_train='2015-12-31'
    start_test='2016-01-01'
    end_test='2016-12-31'
    verbose=False
    features=['MA(20)']
    try:
        
        opts, args = getopt.getopt(argv,"hvs:i:f:",["stock=","investment=","start_train=","end_train=",
                                               "start_test=","end_test=","verbose","features="])
    except getopt.GetoptError:
        print 'trader.py -s <stocks> -i <investment> --start_train <starttraindate> --end_train <endtraindate> --start_test <starttestdat> --end_date <endtestdate> -f <features> -v'
        sys.exit(2)
    for opt, arg in opts:
      if opt == '-h':
          print 'trader.py -s <stocks> -i <investment> --start_train <starttraindate> --end_train <endtraindate> --start_test <starttestdat> --end_date <endtestdate> -f <features> -v'
          print "<stocks> is a list of trading symbols e.g. APPL or 'APPL,GE'" 
          print "features to use - one or more of  ['MA(20)','UB(20)','LB(20)','MA(50)','UB(50)','LB(50)']"
          sys.exit()
      elif opt == '-v':
          verbose = True
      elif opt in ("-s", "--stocks"):
          stocks = arg.split(",")
      elif opt in ("-i", "--invesment"):
          initial_investment = arg
      elif opt == '--start_train':
          start_train = arg
      elif opt == '--end_train':
          end_train = arg
      elif opt == '--start_test':
          start_test = arg
      elif opt == '--end_test':
          end_test = arg
      elif opt == '--features':
          features = arg.split(",")
          

    run(stocks=stocks,initial_investment=initial_investment,start_train=start_train,end_train=end_train,
        start_test=start_test,end_test=end_test,features=features,verbose=verbose)

if __name__ == '__main__':
    main(sys.argv[1:])
