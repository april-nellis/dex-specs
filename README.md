# DEX Specs: A Mean-Field Approach to Decentralized Cryptocurrency Exchanges
This Github repository is published as a companion to the paper [DEX Specs: A Mean-Field Approach to Decentralized Cryptocurrency Exchanges](blank) by Erhan Bayraktar, Asaf Cohen, and April Nellis.

Owner: April Nellis (nellisa@umich.edu)

## Notebooks

### `uniswapAnalysis.ipynb`
This notebook holds the code which performs data analysis on transaction data from Uniswap. It pulls transaction data from Etherscan and price data from CoinGecko.
  - **Key Requirements:** Users must input their own Etherscan API key and CoinGecko API key. Demo keys are free to obtain as of April 12, 2024.
  - **Defaults:** The code is currently set up to pull Uniswap transaction data from September 1-20, 2023 from the ETH/USDC pool with 0.05% fee. Users can easily view transactions for other pools/dates but will have to directly pull all transactions from Etherscan instead of loading the saved CSV file `data/pool3_master_18039179_18245998.csv` which is available for the default dates and pool.
  - **Dependencies:** All necessary packages are loaded in first code cell of notebook.

### `uniswapCalculations.ipynb`
This notebook contains the simulations and analysis of
  1. An N-player Bayesian game among liquidity providers
  2. A mean-field game among liquidity providers
  3. A Stackelberg game between a mean field of liquidity providers as the leader and a MEV Bot performing a Just-In-Time (JIT) liquidity attack as the follower.

  - **Key Requirements:** None.
  - **Defaults:** The code is currently set up to compare simulations with observed data from November 29-30, 2023 from the ETH/USDC pool with 0.05% fee. The liquidity distributions of the pool at two points in time 24 hours apart as saved as `.npy` files in the `data` folder. If users wish to run the code on other dates, they must obtain and format the liquidity distributions themselves.
  - **Dependencies:** All necessary packages are loaded in first code cell of notebook. In addition, the code relies on two `.py` files. Classes and functions are defined in these files to assist in organization. They are:
    1. `DEX.py`
    2. `dexFunction.py`

### `toyDEX.ipynb`
Have questions about the basic mathematics of the concentrated liquidity constant product market maker (CPMM)? Want to try executing swaps and providing liquidity in a small pool that is easier to comprehend? This toy liquidity pool has only 6 ticks and provides code to illustrate examples of liquidity addition and swaps, with visualization.
- **Key Requirements:** None.
- **Defaults:** None.
- **Dependencies:** All necessary packages are loaded in first code cell of notebook. In addition, the code relies on `DEX.py`.

## Python Files

### `DEX.py`
This file defines three classes:
- `Pool`: holds all the functions necessary to implement the concentrated liquidity constant product market maker (CPMM) function for a variety of different liquidity pools.
- `Swapper`: holds the functions necessary to implement a variety of different "semi-naive swappers" for simulations.
- `agentLP`: convenient object to store the type ($\theta$) and position of a liquidity provider.

### `dexFunctions.py`
This file defines functions necessary for calculating the best responses of liquidity providers in various scenarios.
