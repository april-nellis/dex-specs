# define the probability distribution of the exogenous swapper's transactions
import numpy as np
import matplotlib.pyplot as plt

class Swapper:
    def __init__(self, swapper_type, swap_max, swap_min=0):
        self.swapper = swapper_type
        self.seed = 1218274 # a random number, I don't care
        self.swap_max = swap_max
        self.swap_min = swap_min
        self.rng = np.random.default_rng(seed = (self.seed)) # this makes the swaps repeatable at each time

        if swapper_type not in np.arange(7):
            print(f'Invalid swapper type of {swapper_type}.')

    def generateSwaps(self, diff, n_swaps): # diff = p^* - m^*, generates a set of n swaps

        # LOGIC
        # p - m < 0, price of token A is higher in pool so people want to GET token B OUT >>> xi > 0
        # p - m > 0, A is cheaper in pool so people PUT token B IN to get token A >>> xi < 0
        # OPPOSITE SIGNS

        if self.swapper == 0: # uniform swap arrivals
            if diff == 0: # swap could go either way because pool and market equivalent
                x = (self.rng.random(size = n_swaps) - 0.5)*(self.swap_max - self.swap_min)
            elif diff < 0:
                x = self.rng.random(size = n_swaps) * self.swap_min + 1
            elif diff > 0:
                x = self.rng.random(size = n_swaps) * self.swap_max - 1
            else:
                print(f'Invalid diff of {diff}.')

        elif self.swapper == 1: # exponential swap arrivals
            beta = (self.swap_max + self.swap_min)/2 # magnitude of external market orders/swaps

            # probability of swap in each direction
            if diff < 0:
                prob = 0.55
            elif diff > 0:
                prob = 0.45
            elif diff == 0:
                prob = 0.5
            else:
                print(f'Invalid diff of {diff}.')

            x = self.rng.exponential(beta, size = n_swaps) # magnitude of each swap
            sign = 2*self.rng.binomial(1, prob, size = n_swaps) - 1 # make pos 1 or neg 1
            x = x * sign

        elif self.swapper == 2: # discrete distribution swap arrivals
            # probability of swap in each direction
            if diff < 0:
                prob = 0.95
                # either swap_max - 0.2 or -0.2
                x = self.rng.binomial(1, prob, size = n_swaps)*self.swap_max - 1
            elif diff > 0:
                prob = 0.05
                # either swap_min + 0.2 or 0.2
                x = self.rng.binomial(1, prob, size = n_swaps)*self.swap_min + 1
            elif diff == 0:
                prob = 0.5
                # either +avg or -avg
                x = (2*self.rng.binomial(1, prob, size = n_swaps)-1)*(self.swap_min + self.swap_max)/2
            else:
                print(f'Invalid diff of {diff}.')

        elif self.swapper == 3:  # diff = p^* - m^* # mimicking my attempt to fit Uniswap swap arrivals
            prob_size = 0.411
            small = self.swap_min # magnitude of small swaps
            big = self.swap_max # magnitude of big swaps

            # probability of swap in each direction
            if diff > 0:
                prob_sign = 0.466 #0.35
            elif diff < 0:
                prob_sign = 0.567 #0.65
            elif diff == 0:
                prob_sign = 0.5
            else:
                print(f'Invalid diff of {diff}.')

            x_small = self.rng.uniform(0, small, size = n_swaps)
            x_big = self.rng.exponential(big, size = n_swaps) + small # magnitude of each swap

            size = self.rng.binomial(1, prob_size, size = n_swaps) # 0 for big and 1 for small
            x = size*x_small + (1 - size)*x_big

            sign = 2*self.rng.binomial(1, prob_sign, size = n_swaps) - 1 # make pos 1 or neg 1
            x = x * sign

        elif self.swapper == 4:  # diff = p^* - m^* # mimicking lognormal bimodal Uniswap swap arrivals
            prob_size = 0.7 # probability of small swap

            small = self.swap_min # mean of small swaps
            big = self.swap_max # mean of big swaps

            # probability of swap in positive direction
            if diff > 0:
                prob_sign = 0.3#0.466
            elif diff < 0:
                prob_sign = 0.7#0.567
            elif diff == 0:
                prob_sign = 0.5
            else:
                print(f'Invalid diff of {diff}.')

            x_small = self.rng.normal(small, 2, size = n_swaps)
            x_big = self.rng.normal(big, 0.747, size = n_swaps) # magnitude of each swap

            size = self.rng.binomial(1, prob_size, size = n_swaps) # 0 for big and 1 for small
            x = size*np.exp(x_small) + (1 - size)*np.exp(x_big)

            sign = 2*self.rng.binomial(1, prob_sign, size = n_swaps) - 1 # make pos 1 or neg 1
            x = x * sign

        elif self.swapper == 5: # normal swaps with mean swap_max and std. dev. swap_min
            if diff == 0:
                sign = self.rng.choice([1, -1])
            else:
                sign = np.sign(diff) * (-2*self.rng.binomial(1, 0.9, size = n_swaps) + 1)

            x = self.rng.normal(swap_max * sign, swap_min, size = n_swaps)

        elif self.swapper == 6:
            kde = self.swap_max

            points = np.zeros((2, 100))
            domain = np.linspace(-15, 15, 100)
            points[0, :] = diff
            points[1, :] = domain

            cond_pdf = kde(points)
            cond_cdf = np.cumsum(cond_pdf)

            if cond_pdf[-1] == 0:
                print('KDE says impossible')
                x = -1e7*np.sign(diff)
            else:
                cond_cdf = cond_cdf/cond_cdf[-1]

                temp = self.rng.uniform(0, 1)

                if temp <= cond_cdf[0]:
                    x = -1 * np.exp(15)
                elif temp >= cond_cdf[-1]:
                    x = np.exp(15)
                else:
                    i = np.searchsorted(cond_cdf, temp)
                    alpha = (temp - cond_cdf[i-1])/(cond_cdf[i] - cond_cdf[i-1])
                    sample = domain[i-1] + alpha*(domain[i]-domain[i-1])
                    x = np.exp(np.abs(sample))*np.sign(sample)

        else:
            print(f'Invalid swapper type of {self.swapper}.')

        return x

# define the liquidity distribution
class Pool:
    def __init__(self, pool_type, mkt, pool, fee):
        self.type = pool_type

        # list of tick locations: ratio of B/A (number of base per asset)
        if pool_type == 0:
            # very simple
            self.ticks = np.array([0.7, 0.9, 1.1, 1.3])

        elif pool_type == 1:
            # a little less simple
            self.ticks = np.array([1, 1.21, 1.44, 1.69, 1.96, 2.25, 2.56])

        elif type(pool_type) == int and pool_type > 0:
            self.ticks = np.zeros(pool_type)

        else:
            print('Invalid pool type!')

        self.size = len(self.ticks) - 1 # number of intervals
        self.tokA = np.zeros(self.size) # list of amounts of token A in ticks
        self.tokB = np.zeros(self.size) # list of amounts of token B in ticks

        self.pool_er = pool
        self.idx = self.findIndex(self.pool_er) # interval index for pool_er

        self.mkt_er = mkt
        self.mkt_var = 0.00106

        self.fee = fee # amount charged for transaction
        self.gas = 0.002 # for now

        # initialize reserves for K = 1
        if pool_type in [0,1]:
            self.tokA[self.idx] = 1/np.sqrt(self.pool_er) - 1/np.sqrt(self.ticks[self.idx + 1])
            self.tokB[self.idx] = np.sqrt(self.pool_er) - np.sqrt(self.ticks[self.idx])

        # will store range of token A and token B that can be traded in/out
        self.a_ticks = np.zeros(len(self.ticks))
        self.b_ticks = np.zeros(len(self.ticks))
        self.setABTicks()

    ######### FUNCTIONS THAT DO NOT ALTER OBJECT ITSELF ############

    def findIndex(self, p):
        # returns left index aka P_a corresponding to the interval containing price p
        idx = -1
        for i in range(self.size):
            if p >= self.ticks[i] - 1e-3:
                idx = i

        return idx

    def printDetails(self):
        print('-----------------------------------------')
        print(f'The liquidity range is {self.ticks}.')
        print(f'Token A: {self.tokA}.')
        print(f'Token B: {self.tokB}.')
        print(f'The current pool exchange rate is {self.pool_er}. This corresponds to tick index {self.idx}.')
        print(f'Token A swap range: {self.a_ticks}.')
        print(f'Token B swap range: {self.b_ticks}.')
        print('-----------------------------------------')

    def plotLiquidity(self, yLim = [-1, -1]):
        space = max(self.size//20, 1)
        plt.close()
        fig, axs = plt.subplots(2)
        curr_er = self.idx + (self.pool_er - self.ticks[self.idx])/(self.ticks[self.idx + 1] - self.ticks[self.idx]) - 1

        #axs[0].bar(np.arange(self.size), self.tokA, align = 'edge', tick_label = None, width = -0.95, edgecolor='C0', color='none', linewidth=3)
        axs[0].bar(np.arange(self.size), self.tokA, align = 'edge', tick_label = None, width = -1)
        axs[0].axvline(curr_er, color = 'red')
        axs[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        #axs[0].set_xticks(np.arange(-1, self.size, space))
        #axs[0].set_xticklabels(np.round(self.ticks[0::space],2), rotation = 45, ha = 'right')
        #axs[0].set_ylim([0,40])
        axs[0].set_title('Token A Reserves')
        #axs[0].set_xlabel('Exchange Rate')
        axs[0].set_ylabel('Quantity')
        if yLim[0] != -1:
            axs[0].set_ylim(top = yLim[0], bottom = 0)

        #axs[1].bar(np.arange(self.size), self.tokB, align = 'edge', tick_label = None, width = -0.95, edgecolor='C0', color='none', linewidth=3)
        axs[1].bar(np.arange(self.size), self.tokB, align = 'edge', tick_label = None, width = -1)
        #axs[1].set_xticks(np.arange(-1, self.size))
        #axs[1].set_xticklabels(self.ticks)
        axs[1].set_xticks(np.arange(-1, self.size, space))
        axs[1].set_xticklabels(np.round(self.ticks[0::space],2), rotation = 45, ha = 'right')
        #axs[1].set_ylim([0,40])
        axs[1].axvline(curr_er, color = 'red')
        axs[1].set_title('Token B Reserves')
        axs[1].set_xlabel('Exchange Rate')
        axs[1].set_ylabel('Quantity')
        if yLim[1] != -1:
            axs[1].set_ylim(top = yLim[1], bottom = 0)

        #fig.suptitle('Token Distributions')
        plt.show()

    def plotEll(self, ell, yLim = -1):
        space = max(self.size//20, 1)
        plt.close()
        curr_er = self.idx + (self.pool_er - self.ticks[self.idx])/(self.ticks[self.idx + 1] - self.ticks[self.idx]) - 1

        #plt.bar(np.arange(self.size), ell, align = 'edge', tick_label = None, width = -0.95,edgecolor='C0', color='none', linewidth=3)
        plt.bar(np.arange(self.size), ell, align = 'edge', tick_label = None, width = -1)
        plt.xticks(ticks = np.arange(-1, self.size, space), labels = np.round(self.ticks[0::space],2), rotation = 45, ha = 'right')
        plt.axvline(curr_er, color = 'red')
        plt.title('Liquidity Distribution')
        plt.xlabel('Exchange Rate')
        plt.ylabel('Liquidity')
        if yLim != -1:
            plt.ylim(top = yLim, bottom = 0)
        plt.show()

    def calcVirtualReserves(self):
        a = self.tokA[self.idx]
        b = self.tokB[self.idx]
        pa = np.sqrt(self.ticks[self.idx])
        pb = np.sqrt(self.ticks[self.idx + 1])

        p = self.pool_er
        K = self.K(self.idx)

        v_a = a + K/pb
        v_b = b + K*pa

        print(f'Check {self.pool_er} equals {v_b/v_a}.')
        return (v_a, v_b)

    def totalL(self):
        a = np.sum(self.tokA)
        b = np.sum(self.tokB)

        return np.array([a, b]) # total amount of each token

    def resetProfit(self):
        self.profit = np.zeros(2)

    def psi(self, dB, verbose = False):
        # validate that swap is possible
        if not self.validateSwap(dB):
            if verbose: print('Invalid swap')
            return (np.nan, self.pool_er)

        idx = self.idx
        p = self.pool_er

        idx2 = np.searchsorted(self.b_ticks, dB) - 1

        if dB > self.b_ticks[-1] or dB < self.b_ticks[0]:
            if verbose: print('bad!')
            return (np.nan, self.pool_er)
        elif idx2 > idx:
            if verbose: print('positive swap out of interval')
            A = -1*self.a_ticks[idx2]
            dB = dB - self.b_ticks[idx2] # remainder
        elif idx2 < idx:
            if verbose: print('negative swap out of interval')
            A = -1*self.a_ticks[idx2+1]
            dB = dB - self.b_ticks[idx2+1]
        else:
            if verbose: print('staying in current interval')
            A = 0

        K = self.K(idx2)
        pa = np.sqrt(self.ticks[idx2])
        pb = np.sqrt(self.ticks[idx2 + 1])

        top = K**2
        bottom = self.tokB[idx2] + dB + K*pa
        dA = top/bottom - self.tokA[idx2] - K/pb
        A += dA

        newA = self.tokA[idx2] + dA
        newB = self.tokB[idx2] + dB
        new_er = (newB + K*pa)/(newA + K/pb)

        return (A, new_er)

    def validateSwap(self, dB):
        if dB > self.b_ticks[-1] or dB < self.b_ticks[0]:
            return False
        else:
            return True

    def K(self, idx):

        pa = np.sqrt(self.ticks[idx])
        pb = np.sqrt(self.ticks[idx + 1])
        a = self.tokA[idx]
        b = self.tokB[idx]

        if a == 0 and b == 0: # no liquidity
            return 0
        elif a == 0: # all b
            return b/(pb - pa)
        elif b == 0: # all a
            return a/(1/pa - 1/pb)
        else: # mix of a and b
            p = self.pool_er

            if (p/pb - pa) == 0: # weird rare special case
                a2 = 1 - p/(pb**2)
                a1 = -2*b/pb
                a0 = -a*b
                #print(a2, a1, a0)
                K = (-a1 + np.sqrt(a1**2 - 4*a2*a0))/(2*a2) # solve quadratic
                return K
            else:
                return (b - p*a)/(p/pb - pa)

    def maxB(self, idx):
        pa = np.sqrt(self.ticks[idx])
        pb = np.sqrt(self.ticks[idx + 1])

        K = self.K(idx)
        max_b = K*(pb-pa)

        return max_b

    def maxA(self, idx):

        pa = np.sqrt(self.ticks[idx])
        pb = np.sqrt(self.ticks[idx + 1])

        K = self.K(idx)
        max_a = K*(1/pa - 1/pb)

        return max_a

    def copy(self):
        new_nu = Pool(self.type, self.mkt_er, self.pool_er, self.fee)

        new_nu.ticks = np.copy(self.ticks)

        new_nu.size = self.size
        new_nu.idx = self.idx

        new_nu.tokA = np.copy(self.tokA)
        new_nu.tokB = np.copy(self.tokB)

        new_nu.a_ticks = np.copy(self.a_ticks)
        new_nu.b_ticks = np.copy(self.b_ticks)

        return new_nu

    ########### FUNCTIONS THAT DO ALTER THE OBJECT ITSELF ###############

    def setTicks(self, tick_list):
        if len(tick_list) == self.size + 1:
            self.ticks = tick_list
            self.idx = self.findIndex(self.pool_er) # interval index for pool_er
        else:
            print('Size mismatch when setting ticks.')

    def setER(self, rate):
        self.pool_er = rate
        self.idx = self.findIndex(rate)
        self.setABTicks()

    def setMktER(self, rate):
        self.mkt_er = rate

    def setABTicks(self):
        b_ticks = np.zeros(self.size + 1)
        a_ticks = np.zeros(self.size + 1)

        totalB = 0
        totalA = 0

        # first calculate the amount of A you can withdraw when depositing (positive amount of) B
        for i in range(self.idx, self.size):
            pa = np.sqrt(self.ticks[i])
            pb = np.sqrt(self.ticks[i + 1])
            a = self.tokA[i]
            b = self.tokB[i]

            if a+b != 0: # there is actually stuff there

                K = self.K(i)
                max_b = self.maxB(i) # max B this tick can hold if A = 0

                totalB += (max_b - b)
                totalA += a

                b_ticks[i+1] = totalB
                a_ticks[i+1] = totalA

        totalB = 0
        totalA = 0

        # now calculate the amount of A you need to deposit to withdraw (negative amount of) B
        for i in range(self.idx, -1, -1):
            pa = np.sqrt(self.ticks[i])
            pb = np.sqrt(self.ticks[i + 1])
            a = self.tokA[i]
            b = self.tokB[i]

            if a+b != 0: # there is actually stuff there

                K = self.K(i)
                max_a = self.maxA(i) # max A that this tick can hold if B = 0

                totalB += b
                totalA += (max_a - a)

                b_ticks[i] = -totalB
                a_ticks[i] = -totalA

        self.a_ticks = a_ticks
        self.b_ticks = b_ticks


    def calculatePsi(self, showGraph = True):
        # return a graph of the amount of token A you'll get for any specific amount of token B, given the current exchange rate and liquidity distribution
        n = 10 # n determines the 'fine-ness' of the discretization

        b_list = np.empty(0)
        a_list = np.empty(0)

        b_ticks = np.zeros(self.size + 1)
        a_ticks = np.zeros(self.size + 1)

        totalB = 0
        totalA = 0

        if showGraph:
            plt.close()

        # first calculate the amount of A you can withdraw when depositing (positive amount of) B
        for i in range(self.idx, self.size):
            pa = np.sqrt(self.ticks[i])
            pb = np.sqrt(self.ticks[i + 1])
            a = self.tokA[i]
            b = self.tokB[i]

            if a+b != 0: # there is actually stuff there

                K = self.K(i)
                max_b = self.maxB(i) # max B this tick can hold if A = 0
                max_a = self.maxA(i) # max A that this tick can hold if B = 0 (unused?)

                dB = np.linspace(0, max_b-b, n) # amount of B you can contribute in this tick
                dA = a + K/pb - K**2/(b + dB + K*pa) # amount of A you can withdraw for dB

                if np.abs(np.amax(dA) - a) > 0.5: print(f'max dA is {np.amax(dA)} while available A is {a}.')

                dB = dB + totalB
                dA = dA + totalA

                if showGraph:
                    plt.plot(dB, dA, label = f'Exchange rate {self.ticks[i]} to {self.ticks[i+1]}')

                totalB += (max_b - b)
                totalA += a

                b_ticks[i+1] = totalB
                a_ticks[i+1] = totalA

                b_list = np.concatenate((b_list, dB))
                a_list = np.concatenate((a_list, dA))

        totalB = 0
        totalA = 0

        # now calculate the amount of A you need to deposit to withdraw (negative amount of) B
        for i in range(self.idx, -1, -1):
            pa = np.sqrt(self.ticks[i])
            pb = np.sqrt(self.ticks[i + 1])
            a = self.tokA[i]
            b = self.tokB[i]

            if a+b != 0: # there is actually stuff there

                K = self.K(i)
                max_b = self.maxB(i) # max B this tick can hold if A = 0 (unused?)
                max_a = self.maxA(i) # max A that this tick can hold if B = 0

                dA = np.linspace(0, max_a-a, n) # amount of A you can contribute in this tick
                dB = b + K*pa - K**2/(a + dA + K/pb) # amount of B you can withdraw in for dA

                if np.abs(np.amax(dB) - b) > 0.5:
                    print(f'max dB is {np.amax(dB)} while available B is {b} at {i} and A is {a}.')


                dB = dB + totalB
                dA = dA + totalA

                if showGraph:
                    plt.plot(-dB, -dA, label = f'Exchange rate {self.ticks[i]} to {self.ticks[i+1]}')

                totalB += b
                totalA += (max_a - a)

                b_ticks[i] = -totalB
                a_ticks[i] = -totalA

                b_list = np.concatenate((b_list, -dB))
                a_list = np.concatenate((a_list, -dA))

        self.a_ticks = a_ticks
        self.b_ticks = b_ticks

        if showGraph:
            handles, labels = plt.gca().get_legend_handles_labels()

            order = np.zeros(self.size + 1)
            split = self.size-self.idx
            order[:split] = np.flip(np.arange(split))
            order[split:] = np.arange(split, self.size+1)
            order = order.astype(int)

            plt.axvline(0, color = 'black')
            plt.axhline(0, color = 'black')
            plt.xlabel('Amount of Token B deposited into the pool')
            plt.ylabel('Amount of Token A withdrawn from the pool')
            plt.xticks(b_ticks)
            plt.yticks(a_ticks)
            plt.legend([handles[i] for i in order],[labels[i] for i in order])
            plt.show()

        return (a_list, b_list)


    def swap(self, dB, verbose = False):
        idx = self.idx
        dA = 0
        saved_stuff = (self.tokB[idx], self.tokA[idx], self.maxB(idx), self.maxA(idx), self.b_ticks[idx-1:idx+2])
        if dB > 0:
            while self.b_ticks[idx+1] < dB:
                max_b = self.maxB(idx)
                self.tokB[idx] = max_b
                dA -= self.tokA[idx]
                self.tokA[idx] = 0
                idx += 1
                if idx == self.size+1:
                    if verbose: print(f'Invalid swap of size {xi} yet {self.validateSwap(xi)}')
                    return (np.nan, self.pool_er)

            #if idx != self.idx: idx -= 1
            extraB = dB - max(self.b_ticks[idx], 0)
            K = self.K(idx)
            extraA = round(K**2/(extraB + self.tokB[idx] + K*np.sqrt(self.ticks[idx])) - K/np.sqrt(self.ticks[idx+1]) - self.tokA[idx], 9)
            dA += extraA

        else:
            while self.b_ticks[idx] > dB:
                max_a = self.maxA(idx)
                self.tokB[idx] = 0
                dA += (max_a - self.tokA[idx])
                self.tokA[idx] = max_a
                idx -= 1
                if idx == -1:
                    if verbose: print(f'Invalid swap of size {xi} yet {self.validateSwap(xi)}')
                    return (np.nan, self.pool_er)
            #if idx != self.idx: idx += 1
            extraB = dB - min(self.b_ticks[idx+1], 0)
            K = self.K(idx)
            extraA = round(K**2/(extraB + self.tokB[idx] + K*np.sqrt(self.ticks[idx])) - K/np.sqrt(self.ticks[idx+1]) - self.tokA[idx], 9)
            dA += extraA

        self.tokB[idx] += extraB
        self.tokA[idx] += extraA

        if self.tokA[idx] < 0 or self.tokB[idx] < 0:
            print(saved_stuff)
            print(dB, dA, self.idx, idx, self.tokB[idx], self.tokA[idx])

        self.pool_er = round((self.tokB[idx]+ K*np.sqrt(self.ticks[idx]))/(self.tokA[idx] + K/np.sqrt(self.ticks[idx+1])), 9)
        self.b_ticks -= dB
        self.a_ticks += dA
        self.idx = idx

        return dA, self.pool_er

    def updateLiq(self, idx, q): # the units of q are given in the base currency
        # find associated tick index
        dx = -1
        dy = -1
        if self.idx > idx: # desired price is below the current price, contribute B only
            dx = 0
            dy = q
            self.tokB[idx] += dy

        elif self.idx < idx: # desired price is above the current price, contribute A only
            dx = q/self.mkt_er # calculate how many of asset is equivalent to q of the base, given the current exchange rate)
            dy = 0
            self.tokA[idx] += dx

        else: # contribute both in the correct ratio
            pa = np.sqrt(self.ticks[idx])
            pb = np.sqrt(self.ticks[idx + 1])

            p = self.pool_er

            x = self.tokA[idx]
            y = self.tokB[idx]


            if x == 0 and y == 0: # uninitialized tick

                # initialize reserves for K = 1
                if self.ticks[idx+1]-p < 0 and (self.ticks[idx+1]-p)/p > -1e-4:
                    x = 0
                else:
                    x = 1/np.sqrt(p) - 1/pb

                if p-self.ticks[idx] < 0 and (p-self.ticks[idx])/self.ticks[idx] > -1e-4:
                    y = 0
                else:
                    y = np.sqrt(p) - pa



                # find how much to scale up for a total quantity of q
                curr_q = self.mkt_er*x + y
                scale = q/curr_q

                dx = scale * x
                dy = scale * y

                self.tokA[self.idx] = dx
                self.tokB[self.idx] = dy

            else:
                R = y/x

                dx = q/(R + self.mkt_er)
                dy = q - self.mkt_er*dx

                self.tokA[idx] += dx
                self.tokB[idx] += dy

        self.setABTicks() # to update a_ticks and b_ticks
        return dx, dy

    def removeLiq(self, idx, portion):
        a = self.tokA[idx]
        b = self.tokB[idx]

        self.tokA[idx] = (1 - portion)*a
        self.tokB[idx] = (1 - portion)*b
        self.setABTicks() # to update a_ticks and b_ticks

        return (portion*a, portion*b)

    def manualLiqSet(self, idx, qA, qB):
        self.tokA[idx] += qA
        self.tokB[idx] += qB
        self.setABTicks()

    def manualLiqSetAll(self, qAlist, qBlist):
        self.tokA = np.array(qAlist)
        self.tokB = np.array(qBlist)
        self.setABTicks()

    def resetLiquidity(self, l, p_star):
        self.resetProfit()
        self.manualLiqSetAll(np.zeros(self.size), np.zeros(self.size))
        self.setER(p_star)

        self.tokA[:self.idx] = 0
        self.tokB[:self.idx] = l[:self.idx]
        self.tokA[self.idx+1:] = l[self.idx+1:]/self.pool_er
        self.tokB[self.idx+1:] = 0
        self.updateLiq(self.idx, l[self.idx])

        self.setABTicks() # to update a_ticks and b_ticks


class agentLP:
    def __init__(self, pool_size, belief = 0, capital = 1, risk = 1):
        self.a = 0 # lower end of position
        self.b = 0 # upper end of position

        self.lam = risk
        self.K = capital
        self.T = 60*24 # units are minutes
        self.direction = belief # a flag

        # can be K, can be K/(b-a), can be anything
        self.unit_liq = lambda a, b: 0 if (b == a) else self.K/(b - a)

        self.position = np.zeros(pool_size)
        self.position[self.a:self.b] = self.unit_liq(self.a, self.b)

    def agentDetails(self):
        print(f'self.a = {self.a}')
        print(f'self.b = {self.b}')
        print(f'self.lam = {self.lam}')
        print(f'self.K = {self.K}')
        print(f'self.T = {self.T}')
        print(f'self.direction = {self.direction}')
        print(f'self.unit_liq(a, b) = {self.unit_liq(self.a, self.b)}')

    def updateA(self, aNew):
        self.a = aNew
        self.position.fill(0)
        self.position[self.a:self.b] = self.unit_liq(self.a, self.b)

    def updateB(self, bNew):
        self.b = bNew
        self.position.fill(0)
        self.position[self.a:self.b] = self.unit_liq(self.a, self.b)
