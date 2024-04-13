# MFG Functions
import numpy as np
from DEX import Pool, Swapper, agentLP
import matplotlib.pyplot as plt
import os.path
import os
import time

# Parameters
capital_list = np.array([2124, 35786, 1706034])

def pretty_rounding(x):
    if abs(int(x) - x) < 1e-3:
        return int(x)
    else:
        return round(x, 2)

def totalFees(nu, xi, bot_zeta = 0):
        total_fee = np.zeros(nu.size)
        bot_profit = 0
        portion = 0
        q = 2e7
        curr_idx = nu.idx

        if xi == 0:
            print('No fees earned!')
            return np.zeros(nu.size)

        if bot_zeta > 0 and bot_zeta <= 1:
            flag = np.random.binomial(n=1, p=bot_zeta) # probability of bot attack
        elif bot_zeta == 0:
            flag = 0
        else:
            print('bad zeta')

        if flag == 1:
            if xi > 0:
                if nu.tokA[nu.idx] > 0:
                    j = nu.idx
                else:
                    j = nu.idx + 1

                portion = (q/nu.pool_er)/((q/nu.pool_er) + nu.tokA[j])

            else:
                if nu.tokB[nu.idx] > 0:
                    j = nu.idx
                else:
                    j = nu.idx - 1
                portion = q/(q + nu.tokB[j])

            L = portion * nu.K(j)
            if L > 0:
                threshold = botThreshold(nu, L)
            else:
                threshold = np.array([-1e6, 1e6])

            if xi > threshold[0] and xi < threshold[1]:
                portion = 0
            else:
                #print(xi, threshold, 'bot attack')
                nu.updateLiq(j, q)

        if nu.validateSwap(xi):
            initB = np.copy(nu.tokB)
            b, p2 = nu.swap(xi, verbose = True)
            newB = np.copy(nu.tokB)
            total_fee = np.abs(newB - initB) * nu.fee * (1 - portion)
            bot_profit = np.abs(xi) * nu.fee * portion

        if flag == 1 and portion > 0:
            nu.removeLiq(j, portion)

        return total_fee, bot_profit

def V(direction, T, nu, vBot = 0, M = 100):
    rng = np.random.default_rng(12345)

    swapper = Swapper(swapper_type = 4, swap_max = 11.764, swap_min = 7.446)
    #swapper = Swapper(swapper_type = 5, swap_max = agent.mu, swap_min = agent.sigma)

    fees = np.zeros((M, nu.size))
    bot = np.zeros(M)

    ell0 = nu.tokA * nu.pool_er + nu.tokB
    p0 = nu.pool_er

    reset_time = 0
    calc_time = 0
    findx1x2_time = 0
    big_xi_count = 0

    sigma = 0.00106

    if direction == 1:
        mu = 0.0001
        #shock_list = [1, 1, 1, 1, 1, 1, 1, 1.005, 1/1.003] #[1, 1, 1.01, 1.001, 1.025, 1/1.01, 1/1.001, 1/1.005]
    elif direction == 0:
        mu = 0
        #shock_list = [1, 1, 1, 1, 1, 1, 1.003, 1/1.003, 1.001, 1/1.001, 1.005, 1/1.005] #[1, 1, 1.01, 1.001, 1.005, 1/1.01, 1/1.001, 1/1.02]
    else:
        mu = -0.0001
        #shock_list = [1, 1, 1, 1, 1, 1, 1, 1.001, 1/1.003] #[1, 1, 1.01, 1.001, 1.015, 1/1.01, 1/1.001, 1/1.015]

    for m in range(M):

        if m%10 == 0:
            print(f'{m} paths done.')
        nu2 = nu.copy()
        penalty_count = 0
        for t in range(T*5): # Five blocks per minute (12 seconds apart)
            diff = nu2.pool_er - nu2.mkt_er
            xi = swapper.generateSwaps(diff, 1)[0]

            fees_t, bot_t = totalFees(nu2, xi, bot_zeta = vBot)
            bot[m] += bot_t

            if np.isnan(fees_t).any():
                penalty_count += 1
            else:
                fees[m] += fees_t

            if t%25 == 0: # every 5 minutes
                dW = rng.normal(0, 1)
                nu2.mkt_er = nu2.mkt_er * np.exp(mu - sigma**2 + sigma*dW) #nu2.mkt_er * shock

    return fees, bot

def cdf_approx(aFees, aNu):
    cdf = np.zeros((aFees.shape[0], aNu.size, 3))

    for j in range(aFees.shape[0]):
        for i in range(aNu.size):
            cdf[j, i, 0] = aNu.b_ticks[i]
            cdf[j, i, 1] = aNu.b_ticks[i + 1]
            cdf[j, i, 2] = aFees[j, i]

        cdf[j, :, 2] = cdf[j, :, 2]/np.sum(cdf[j, :, 2]) * 2e8 # avg 24 hour voluem is abt 200m USD
    return cdf

def V_approx(b_ticks, cdf): # cdf shape is (trials, CDF entries, 3) where last dimension is (lower bound, upper bound, volume)
    ticks = b_ticks
    nu_size = len(b_ticks)-1
    fees = np.zeros((cdf.shape[0], nu_size))
    cdf_entries = cdf[0, :,:2] # should be idential along first axis, so choose first element

    nu_idx = 0

    for i in range(len(cdf_entries)):
        lower = cdf_entries[i, 0]
        upper = cdf_entries[i, 1]

        #get to correct index
        if upper < ticks[nu_idx]:
            continue

        while lower > ticks[nu_idx]:
            nu_idx += 1

        #distribute appropriately
        if nu_idx > 0:
            fees[:, nu_idx-1] += (ticks[nu_idx] - lower)/(upper - lower) * cdf[:, i, 2]

        nu_idx += 1

        while nu_idx <= nu_size and ticks[nu_idx] < upper:
            fees[:, nu_idx-1] += (ticks[nu_idx] - ticks[nu_idx-1])/(upper-lower) * cdf[:, i, 2]
            nu_idx += 1


        if nu_idx <= nu_size and ticks[nu_idx] >= upper:
            fees[:, nu_idx-1] += (upper - ticks[nu_idx-1])/(upper - lower) * cdf[:, i, 2]

        if nu_idx >= nu_size:
            break

    return fees

def br_N(agent, actions, samples, cdf, aNu): # choose BR given (unknown) actions of all other players
    best_action = -1
    j1 = -1
    j2 = -1
    my_max = -1e8

    # only check the actions which made the cut
    for i in range(actions.shape[1]):
        #if i%50 == 0: print(f'action iteration: {i}')
        curri = np.argmax(actions[:,i] > 0) # first nonzero index
        currj = aNu.size - np.argmax(np.flip(actions[:,i]) > 0) # one after last nonzero index
        curr_val = exp_payoff(curri, currj, agent, actions, samples, cdf, aNu)

        if curr_val > my_max:
            my_max = curr_val
            best_action = i
            j1 = curri
            j2 = currj

    print(f'action {best_action} = {(j1, j2)}')
    return best_action, my_max

# calculate expected payoff for position associated with (curri, currj)
def exp_payoff(i, j, agent, actions, samples, cdf, aNu):
    scale = (np.sqrt(aNu.ticks[aNu.idx+1:]) - np.sqrt(aNu.ticks[aNu.idx:-1]))/(np.sqrt(1/aNu.ticks[aNu.idx:-1]) - np.sqrt(1/aNu.ticks[aNu.idx+1:]))/aNu.pool_er

    expV = 0

    for theta in range(samples.shape[0]): # MC samples (100)
        curr_ell = np.sum(actions[:,samples[theta].astype(int)], axis = 1)*100

        my_ell = np.zeros(aNu.size)
        my_ell[i:j] += agent.K/(j-i)
        curr_ell += my_ell

        curr_ell += 1
        portion = np.divide(my_ell, curr_ell, out=np.zeros(aNu.size), where=curr_ell!=0) # profit as portion of liq

        curr_ell[aNu.idx:] *= scale
        b_list = np.cumsum(curr_ell)
        b_list = np.concatenate(([0], b_list))
        b_list -= b_list[aNu.idx]

        fees = V_approx(b_list, cdf[agent.direction+1]) # approximating V (with cdf which is also an approximation lol), but faster

        my_fees = np.sum(fees*portion, axis = 1)

        mean = np.mean(my_fees)
        var = np.std(my_fees)
        currV = mean - agent.lam*var
        expV += currV

    expV = expV/samples.shape[0]

    return expV

def optimize_N(agent, nu, cdf): # choose BR given (known) actions of all other players
    profit = np.zeros(nu.size)

    p0 = nu.pool_er
    ell0 = nu.pool_er * nu.tokA + nu.tokB

    reset_time = 0
    calc_time = 0
    nfindx1x2_time = 0

    a_star = agent.a
    b_star = agent.b
    vmax = 11

    if agent.lam < 0:
        agent.a = 0
        agent.b = nu.size
        return (0, nu.size, vmax)

    trend = int(agent.direction + 1)

    ell1 = ell0 - agent.position # everyone else's liquidity

    a_range = np.arange(0, nu.idx, 10)
    if a_range[-1] != nu.idx:
        a_range = np.concatenate((a_range, [nu.idx]))

    b_range = np.arange(nu.idx+1, nu.size, 10)
    if b_range[-1] != nu.size:
        b_range = np.concatenate((b_range, [nu.size]))

    for i in a_range:
        for j in b_range:
            nu2 = nu.copy()

            ell2 = np.copy(ell1)
            ell2[i:j] += agent.unit_liq(i,j)
            nu2.resetLiquidity(ell2, nu2.pool_er)

            fees = V_approx(nu2, cdf[trend])
            portion = np.divide((ell2 - ell1), ell2, out=np.zeros(nu.size), where=ell2!=0) # profit as portion of liq

            mean = np.sum(np.mean(fees * portion, axis = 1))
            var = np.matmul(np.matmul(portion, np.cov(np.transpose(fees))), portion)
            profit = mean - agent.lam * var

            if i == a_range[0] and j == b_range[0]:
                vmax = profit
                a_star = i
                b_star = j

            elif profit > vmax:
                vmax = profit
                a_star = i
                b_star = j

    agent.updateA(a_star)
    agent.updateB(b_star)

    return (a_star, b_star, vmax)

def findEll(length, agent_list):
    ell = np.zeros(length)
    for agent in agent_list:
        ell[agent.a:agent.b] += agent.unit_liq(agent.a, agent.b)
    ell += 1e-2 # I think this is to prevent shrinkage
    return ell

# Fictitious play algorithm for N-player game where types are already assigned
def equilibriumNP(nu, agents, cdfT):
    error = 100
    error_prev = error*2

    ell_curr = findEll(nu.size, agents)

    p_star = nu.pool_er
    nu.resetLiquidity(ell_curr, p_star)

    big_count = 0
    while error > 1 and big_count < 30:
        big_count += 1
        count = 0
        for ag in agents:
            if ag.lam > -0.5:
                nu.resetLiquidity(ell_curr, p_star)
                a, b, v = optimize_N(ag, nu, cdfT)
            else:
                a,b,v = 0, nu.size, 0
            count += 1
            print(f'Agent {count} chose {(a, b)} with profit {v}')

        ell_prev = ell_curr
        ell_curr = findEll(nu.size, agents)

        error_prev = error
        error = np.mean(np.square(ell_curr - ell_prev))
        print(f'error is {error}')

        if error_prev == error:
            print('Looping, uh oh!')
            break

    print('Done!')
    return ell_curr


def mean_var_opt(agent, fees, nu):

    if agent.lam < 0: # infinitely risk-avers
        return 0, nu.size, 0, 0

    ell0 = nu.pool_er * nu.mkt_er + nu.tokB

    unit_fees = np.divide(fees, ell0, out=np.zeros_like(fees), where=ell0!=0)
    pi = np.mean(unit_fees, axis = 0)
    my_cov = np.cov(np.transpose(unit_fees))
    #pi = np.mean(fees, axis =0)

    min_idx = 0#min(np.argmax(pi > 0), nu.idx-1)
    max_idx = nu.size#max(len(pi) - np.argmax(np.flip(pi) > 0), nu.idx + 1)


    if agent.a == 0 and agent.b == 0: # initial state
        #print('initial state')
        g = 0
        a_star = min_idx
        b_star = max_idx
    else:
        g = 0
        a_star = agent.a
        b_star = agent.b


    my_max = np.sum(pi[a_star:b_star]) * agent.unit_liq(a_star, b_star) - agent.lam * np.sum(my_cov[a_star:b_star, a_star:b_star]) * (agent.K/(b_star - a_star))**2
    initial = my_max

    #print(pi[a_star:b_star])
    #print(min_idx, max_idx, my_max)

    for a in range(nu.size):
        for b in range(a+1, nu.size+1):
            curr = np.sum(pi[a:b])*agent.K/(b - a) - agent.lam * np.sum(my_cov[a:b, a:b]) * (agent.K/(b - a))**2
            if curr > my_max:
                my_max = curr
                a_star = a
                b_star = b

    if my_max - g < initial: # worse off to adjust than to stay
        a_star = agent.a
        b_star = agent.b
        my_max = initial
    else:
        agent.a = a_star
        agent.b = b_star


    return a_star, b_star, my_max, np.mean(fees)

def mfg_equilibrium(agent, nu):

    ell_curr = nu.tokA * nu.pool_er + nu.tokB
    p_star = nu.pool_er

    v = np.sum(ell_curr) # total value locked in pool

    max_iter = 50
    error = v**2
    background = 0 # 1e-4 * v # ensure at least 0.01% of liquidity in each interval

    count = 0
    while error > background and count < max_iter:
        print(f'---------------- ITERATION {count} -------------------')
        nu.resetLiquidity(ell_curr, p_star)
        #nu.printDetails()
        ell_prev = ell_curr
        ell_curr = agent.mfg_br(nu, M = 500)
        count += 1

        #print(ell_prev)
        print(ell_curr[ell_curr > background])

        error_prev = error
        error = np.mean(np.abs(ell_curr - ell_prev))
        print(f'error is {error}')

        # need to avoid loops -> there's a wider question of when an equilibrium really exists
        if error_prev == error:
            print('Looping, uh oh!')
            break

    print('Done!')

    if count == max_iter:
        print('Reached max iterations without converging.')

    return ell_curr

def botThreshold(nu, L):
    G = 27.6 # gas fees

    l = nu.K(nu.idx)

    D_p = ((l + L)/l)*((1 + nu.fee)*nu.pool_er - nu.mkt_er)*(nu.tokA[nu.idx] + l/np.sqrt(nu.ticks[nu.idx+1])) - G*(l + L)/L
    D_m = ((l + L)/l)*((1 - nu.fee)*nu.pool_er - nu.mkt_er)*(nu.tokA[nu.idx] + l/np.sqrt(nu.ticks[nu.idx+1])) - G*(l + L)/L

    E = -(nu.tokB[nu.idx] + l*np.sqrt(nu.ticks[nu.idx])) * G*(l + L)**2/(L*l)

    xi_plus = max((-D_p + np.sqrt(D_p**2 - 4*(1 + nu.fee)*E))/(2 + 2*nu.fee), 0)
    xi_minus = min((-D_m - np.sqrt(D_m**2 - 4*(1 - nu.fee)*E))/(2 - 2*nu.fee), 0)

    pa = np.sqrt(nu.ticks[nu.idx])
    pb = np.sqrt(nu.ticks[nu.idx + 1])

    if xi_plus > (l + L)*(pb-pa) - (l + L)*nu.tokB[nu.idx]/l:
        xi_plus = np.inf
    if xi_minus < -(l + L)*nu.tokB[nu.idx]/l:
        xi_minus = -np.inf

    return np.array([xi_minus, xi_plus])

# might want to delete, idk

def simulate_Stackelberg(aNu, simZeta):
    M = 100

    bot_pi = np.zeros(M)
    lp_pi = np.zeros(M)

    bot_a = 0
    bot_b = 0

    p_star = 0

    #swapper = Swapper(swapper_type = 3, swap_max = 52113, swap_min = 5000)
    swapper = Swapper(swapper_type = 4, swap_max = 11.764, swap_min = 7.446)

    shock_list = [1, 1, 1.05, 0.95]

    rng2 = np.random.default_rng(2468)

    T = 60 # units in minutes

    ell = aNu.tokA * aNu.mkt_er + aNu.tokB
    for m in range(M):
        aNu2 = aNu.copy()
        for t in range(T*5):
            if t == 3:
                shock = 1#rng2.choice(shock_list)
                aNu2.mkt_er = aNu2.mkt_er * shock

            if np.abs(aNu2.mkt_er - aNu2.pool_er) > 5:
                prob = 0.7
            else:
                prob = 0.5

            num_swaps = rng2.binomial(n = 3, p = prob) + 1
            #print(f'------ at t = {t}, numswaps = {num_swaps}')
            xi_list = swapper.generateSwaps(aNu2.pool_er - aNu2.mkt_er, num_swaps)

            for xi in xi_list:
                fee_curr, bot_curr = totalFees(aNu2, xi, bot_zeta = simZeta)

                lp_pi[m] += np.sum(fee_curr)
                bot_pi[m] += bot_curr
            # end of xi loop
        p_star += aNu2.pool_er
        # end of time loop
    # end of repeats

    bot_holdings = bot_a * aNu2.mkt_er + bot_b

    #print(f"Bot start: (0, 0), bot end: {bot_a, bot_b} -> value of {bot_holdings}")
    print(f"Bot profit: {np.mean(bot_pi)} w std.dev {np.std(bot_pi)}")
    print(f"LP total profit: {np.mean(lp_pi)} w std.dev {np.std(lp_pi)}")
    print(f"Exchange rate difference: pool {p_star/M} vs. market {aNu2.mkt_er}")

    return bot_pi, lp_pi

def calibrateAgents(nu, zeta, myT, myM, agent_weights = None, fees_in = None, loadFile = True):
    fileName = f'data/fees_zeta_{int(zeta*10)}.npy'
    ell0 = nu.tokB + nu.pool_er * nu.tokA
    if fees_in is None:
        if os.path.isfile(fileName) and loadFile == True:
            print('Loading fees')
            fees = np.load(fileName)
            print(fees.shape)
        else:
            print('Calculating fees')
            start = time.time()
            fees1, bot = V(-1, myT, nu, vBot = zeta, M=myM)
            fees2, bot = V(0, myT, nu, vBot = zeta, M=myM)
            fees3, bot = V(1, myT, nu, vBot = zeta, M=myM)
            fees = np.stack([fees1, fees2, fees3], axis = 0)

            n=2
            while os.path.exists(f'{fileName}.txt'):
                fileName = fileName + f'v{n}'
                n+=1

            np.save(fileName, fees)
            print(f'Path simulation time: {time.time() - start}')
    else:
        fees = fees_in
    start = time.time()
    lambda_list = np.concatenate([np.linspace(0, 0.09, 10), np.linspace(0.1, 1.9, 19), np.arange(2, 6, 0.5)])
    #capital_list = np.array([2682, 17575, 232995]) #np.array([1e3, 1e4, 1e5])
    my_dict = np.zeros((len(lambda_list)*3*len(capital_list) + 1, 5)) # 5D = [position_lower, position_upper, k lambda, belief]
    my_dict[0] = np.array([0,nu.size,5e4, -1, 0])
    count = 1
    agent1 = agentLP(nu.size, belief = 0, capital = 5e4, risk = 0)

    for i in lambda_list: # these are risk levels
        for j in np.arange(3):
            for k in capital_list:
                agent1 = agentLP(nu.size, belief = j-1, capital = k, risk = i)
                a, b, c, d = mean_var_opt(agent1, fees[j], nu)
                my_dict[count] = np.array([a, b, k, i, j-1]) # [j1, j2, k, lambda, delta]
                count += 1

    my_dict = my_dict[my_dict[:,1] > 0] # remove elements where optimal position is (0,0), probably unnecessary
    print(f'Calibration time: {time.time()-start}')
    return fees, my_dict
