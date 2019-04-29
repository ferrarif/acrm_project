'''
Created on 15 Feb 2019

@author: francescoferrari
'''

sp500 = [
            ('A','Health Care','USD'),
            ('AAL','Industrials','USD'),
            ('AAP','Consumer Discretionary','USD'),
            ('AAPL','Information Technology','USD'),
            ('ABBV','Health Care','USD'),
            ('ABC','Health Care','USD'),
            ('ABMD','Health Care','USD'),
            ('ABT','Health Care','USD'),
            ('ACN','Information Technology','USD'),
            ('ADBE','Information Technology','USD'),
            ('ADI','Information Technology','USD'),
            ('ADM','Consumer Staples','USD'),
            ('ADP','Information Technology','USD'),
            ('ADS','Information Technology','USD'),
            ('ADSK','Information Technology','USD'),
            ('AEE','Utilities','USD'),
            ('AEP','Utilities','USD'),
            ('AES','Utilities','USD'),
            ('AFL','Financials','USD'),
            ('AGN','Health Care','USD'),
            ('AIG','Financials','USD'),
            ('AIV','Real Estate','USD'),
            ('AIZ','Financials','USD'),
            ('AJG','Financials','USD'),
            ('AKAM','Information Technology','USD'),
            ('ALB','Materials','USD'),
            ('ALGN','Health Care','USD'),
            ('ALK','Industrials','USD'),
            ('ALL','Financials','USD'),
            ('ALLE','Industrials','USD'),
            ('ALXN','Health Care','USD'),
            ('AMAT','Information Technology','USD'),
            ('AMD','Information Technology','USD'),
            ('AME','Industrials','USD'),
            ('AMG','Financials','USD'),
            ('AMGN','Health Care','USD'),
            ('AMP','Financials','USD'),
            ('AMT','Real Estate','USD'),
            ('AMZN','Consumer Discretionary','USD'),
            ('ANET','Information Technology','USD'),
            ('ANSS','Information Technology','USD'),
            ('ANTM','Health Care','USD'),
            ('AON','Financials','USD'),
            ('AOS','Industrials','USD'),
            ('APA','Energy','USD'),
            ('APC','Energy','USD'),
            ('APD','Materials','USD'),
            ('APH','Information Technology','USD'),
            ('APTV','Consumer Discretionary','USD'),
            ('ARE','Real Estate','USD'),
            ('ARNC','Industrials','USD'),
            ('ATVI','Communication Services','USD'),
            ('AVB','Real Estate','USD'),
            ('AVGO','Information Technology','USD'),
            ('AVY','Materials','USD'),
            ('AWK','Utilities','USD'),
            ('AXP','Financials','USD'),
            ('AZO','Consumer Discretionary','USD'),
            ('BA','Industrials','USD'),
            ('BAC','Financials','USD'),
            ('BAX','Health Care','USD'),
            ('BBT','Financials','USD'),
            ('BBY','Consumer Discretionary','USD'),
            ('BDX','Health Care','USD'),
            ('BEN','Financials','USD'),
            ('BF-B','Consumer Staples','USD'),
            ('BHF','Financials','USD'),
            ('BHGE','Energy','USD'),
            ('BIIB','Health Care','USD'),
            ('BK','Financials','USD'),
            ('BKNG','Consumer Discretionary','USD'),
            ('BLK','Financials','USD'),
            ('BLL','Materials','USD'),
            ('BMY','Health Care','USD'),
            ('BR','Information Technology','USD'),
            ('BRK-B','Financials','USD'),
            ('BSX','Health Care','USD'),
            ('BWA','Consumer Discretionary','USD'),
            ('BXP','Real Estate','USD'),
            ('C','Financials','USD'),
            ('CAG','Consumer Staples','USD'),
            ('CAH','Health Care','USD'),
            ('CAT','Industrials','USD'),
            ('CB','Financials','USD'),
            ('CBOE','Financials','USD'),
            ('CBRE','Real Estate','USD'),
            ('CBS','Communication Services','USD'),
            ('CCI','Real Estate','USD'),
            ('CCL','Consumer Discretionary','USD'),
            ('CDNS','Information Technology','USD'),
            ('CE','Materials','USD'),
            ('CELG','Health Care','USD'),
            ('CERN','Health Care','USD'),
            ('CF','Materials','USD'),
            ('CFG','Financials','USD'),
            ('CHD','Consumer Staples','USD'),
            ('CHRW','Industrials','USD'),
            ('CHTR','Communication Services','USD'),
            ('CI','Health Care','USD'),
            ('CINF','Financials','USD'),
            ('CL','Consumer Staples','USD'),
            ('CLX','Consumer Staples','USD'),
            ('CMA','Financials','USD'),
            ('CMCSA','Communication Services','USD'),
            ('CME','Financials','USD'),
            ('CMG','Consumer Discretionary','USD'),
            ('CMI','Industrials','USD'),
            ('CMS','Utilities','USD'),
            ('CNC','Health Care','USD'),
            ('CNP','Utilities','USD'),
            ('COF','Financials','USD'),
            ('COG','Energy','USD'),
            ('COO','Health Care','USD'),
            ('COP','Energy','USD'),
            ('COST','Consumer Staples','USD'),
            ('COTY','Consumer Staples','USD'),
            ('CPB','Consumer Staples','USD'),
            ('CPRI','Consumer Discretionary','USD'),
            ('CPRT','Industrials','USD'),
            ('CRM','Information Technology','USD'),
            ('CSCO','Information Technology','USD'),
            ('CSX','Industrials','USD'),
            ('CTAS','Industrials','USD'),
            ('CTL','Communication Services','USD'),
            ('CTSH','Information Technology','USD'),
            ('CTXS','Information Technology','USD'),
            ('CVS','Health Care','USD'),
            ('CVX','Energy','USD'),
            ('CXO','Energy','USD'),
            ('D','Utilities','USD'),
            ('DAL','Industrials','USD'),
            ('DE','Industrials','USD'),
            ('DFS','Financials','USD'),
            ('DG','Consumer Discretionary','USD'),
            ('DGX','Health Care','USD'),
            ('DHI','Consumer Discretionary','USD'),
            ('DHR','Health Care','USD'),
            ('DIS','Communication Services','USD'),
            ('DISCA','Communication Services','USD'),
            ('DISCK','Communication Services','USD'),
            ('DISH','Communication Services','USD'),
            ('DLR','Real Estate','USD'),
            ('DLTR','Consumer Discretionary','USD'),
            ('DOV','Industrials','USD'),
            ('DRE','Real Estate','USD'),
            ('DRI','Consumer Discretionary','USD'),
            ('DTE','Utilities','USD'),
            ('DUK','Utilities','USD'),
            ('DVA','Health Care','USD'),
            ('DVN','Energy','USD'),
            ('DWDP','Materials','USD'),
            ('DXC','Information Technology','USD'),
            ('EA','Communication Services','USD'),
            ('EBAY','Consumer Discretionary','USD'),
            ('ECL','Materials','USD'),
            ('ED','Utilities','USD'),
            ('EFX','Industrials','USD'),
            ('EIX','Utilities','USD'),
            ('EL','Consumer Staples','USD'),
            ('EMN','Materials','USD'),
            ('EMR','Industrials','USD'),
            ('EOG','Energy','USD'),
            ('EQIX','Real Estate','USD'),
            ('EQR','Real Estate','USD'),
            ('ES','Utilities','USD'),
            ('ESS','Real Estate','USD'),
            ('ETFC','Financials','USD'),
            ('ETN','Industrials','USD'),
            ('ETR','Utilities','USD'),
            ('EVRG','Utilities','USD'),
            ('EW','Health Care','USD'),
            ('EXC','Utilities','USD'),
            ('EXPD','Industrials','USD'),
            ('EXPE','Consumer Discretionary','USD'),
            ('EXR','Real Estate','USD'),
            ('F','Consumer Discretionary','USD'),
            ('FANG','Energy','USD'),
            ('FAST','Industrials','USD'),
            ('FB','Communication Services','USD'),
            ('FBHS','Industrials','USD'),
            ('FCX','Materials','USD'),
            ('FDX','Industrials','USD'),
            ('FE','Utilities','USD'),
            ('FFIV','Information Technology','USD'),
            ('FIS','Information Technology','USD'),
            ('FISV','Information Technology','USD'),
            ('FITB','Financials','USD'),
            ('FL','Consumer Discretionary','USD'),
            ('FLIR','Information Technology','USD'),
            ('FLR','Industrials','USD'),
            ('FLS','Industrials','USD'),
            ('FLT','Information Technology','USD'),
            ('FMC','Materials','USD'),
            ('FOX','Communication Services','USD'),
            ('FOXA','Communication Services','USD'),
            ('FRC','Financials','USD'),
            ('FRT','Real Estate','USD'),
            ('FTI','Energy','USD'),
            ('FTNT','Information Technology','USD'),
            ('FTV','Industrials','USD'),
            ('GD','Industrials','USD'),
            ('GE','Industrials','USD'),
            ('GILD','Health Care','USD'),
            ('GIS','Consumer Staples','USD'),
            ('GLW','Information Technology','USD'),
            ('GM','Consumer Discretionary','USD'),
            ('GOOG','Communication Services','USD'),
            ('GOOGL','Communication Services','USD'),
            ('GPC','Consumer Discretionary','USD'),
            ('GPN','Information Technology','USD'),
            ('GPS','Consumer Discretionary','USD'),
            ('GRMN','Consumer Discretionary','USD'),
            ('GS','Financials','USD'),
            ('GT','Consumer Discretionary','USD'),
            ('GWW','Industrials','USD'),
            ('HAL','Energy','USD'),
            ('HAS','Consumer Discretionary','USD'),
            ('HBAN','Financials','USD'),
            ('HBI','Consumer Discretionary','USD'),
            ('HCA','Health Care','USD'),
            ('HCP','Real Estate','USD'),
            ('HD','Consumer Discretionary','USD'),
            ('HES','Energy','USD'),
            ('HFC','Energy','USD'),
            ('HIG','Financials','USD'),
            ('HII','Industrials','USD'),
            ('HLT','Consumer Discretionary','USD'),
            ('HOG','Consumer Discretionary','USD'),
            ('HOLX','Health Care','USD'),
            ('HON','Industrials','USD'),
            ('HP','Energy','USD'),
            ('HPE','Information Technology','USD'),
            ('HPQ','Information Technology','USD'),
            ('HRB','Consumer Discretionary','USD'),
            ('HRL','Consumer Staples','USD'),
            ('HRS','Industrials','USD'),
            ('HSIC','Health Care','USD'),
            ('HST','Real Estate','USD'),
            ('HSY','Consumer Staples','USD'),
            ('HUM','Health Care','USD'),
            ('IBM','Information Technology','USD'),
            ('ICE','Financials','USD'),
            ('IDXX','Health Care','USD'),
            ('IFF','Materials','USD'),
            ('ILMN','Health Care','USD'),
            ('INCY','Health Care','USD'),
            ('INFO','Industrials','USD'),
            ('INTC','Information Technology','USD'),
            ('INTU','Information Technology','USD'),
            ('IP','Materials','USD'),
            ('IPG','Communication Services','USD'),
            ('IPGP','Information Technology','USD'),
            ('IQV','Health Care','USD'),
            ('IR','Industrials','USD'),
            ('IRM','Real Estate','USD'),
            ('ISRG','Health Care','USD'),
            ('IT','Information Technology','USD'),
            ('ITW','Industrials','USD'),
            ('IVZ','Financials','USD'),
            ('JBHT','Industrials','USD'),
            ('JCI','Industrials','USD'),
            ('JEC','Industrials','USD'),
            ('JEF','Financials','USD'),
            ('JKHY','Information Technology','USD'),
            ('JNJ','Health Care','USD'),
            ('JNPR','Information Technology','USD'),
            ('JPM','Financials','USD'),
            ('JWN','Consumer Discretionary','USD'),
            ('K','Consumer Staples','USD'),
            ('KEY','Financials','USD'),
            ('KEYS','Information Technology','USD'),
            ('KHC','Consumer Staples','USD'),
            ('KIM','Real Estate','USD'),
            ('KLAC','Information Technology','USD'),
            ('KMB','Consumer Staples','USD'),
            ('KMI','Energy','USD'),
            ('KMX','Consumer Discretionary','USD'),
            ('KO','Consumer Staples','USD'),
            ('KR','Consumer Staples','USD'),
            ('KSS','Consumer Discretionary','USD'),
            ('KSU','Industrials','USD'),
            ('L','Financials','USD'),
            ('LB','Consumer Discretionary','USD'),
            ('LEG','Consumer Discretionary','USD'),
            ('LEN','Consumer Discretionary','USD'),
            ('LH','Health Care','USD'),
            ('LIN','Materials','USD'),
            ('LKQ','Consumer Discretionary','USD'),
            ('LLL','Industrials','USD'),
            ('LLY','Health Care','USD'),
            ('LMT','Industrials','USD'),
            ('LNC','Financials','USD'),
            ('LNT','Utilities','USD'),
            ('LOW','Consumer Discretionary','USD'),
            ('LRCX','Information Technology','USD'),
            ('LUV','Industrials','USD'),
            ('LW','Consumer Staples','USD'),
            ('LYB','Materials','USD'),
            ('M','Consumer Discretionary','USD'),
            ('MA','Information Technology','USD'),
            ('MAA','Real Estate','USD'),
            ('MAC','Real Estate','USD'),
            ('MAR','Consumer Discretionary','USD'),
            ('MAS','Industrials','USD'),
            ('MAT','Consumer Discretionary','USD'),
            ('MCD','Consumer Discretionary','USD'),
            ('MCHP','Information Technology','USD'),
            ('MCK','Health Care','USD'),
            ('MCO','Financials','USD'),
            ('MDLZ','Consumer Staples','USD'),
            ('MDT','Health Care','USD'),
            ('MET','Financials','USD'),
            ('MGM','Consumer Discretionary','USD'),
            ('MHK','Consumer Discretionary','USD'),
            ('MKC','Consumer Staples','USD'),
            ('MLM','Materials','USD'),
            ('MMC','Financials','USD'),
            ('MMM','Industrials','USD'),
            ('MNST','Consumer Staples','USD'),
            ('MO','Consumer Staples','USD'),
            ('MOS','Materials','USD'),
            ('MPC','Energy','USD'),
            ('MRK','Health Care','USD'),
            ('MRO','Energy','USD'),
            ('MS','Financials','USD'),
            ('MSCI','Financials','USD'),
            ('MSFT','Information Technology','USD'),
            ('MSI','Information Technology','USD'),
            ('MTB','Financials','USD'),
            ('MTD','Health Care','USD'),
            ('MU','Information Technology','USD'),
            ('MXIM','Information Technology','USD'),
            ('MYL','Health Care','USD'),
            ('NBL','Energy','USD'),
            ('NCLH','Consumer Discretionary','USD'),
            ('NDAQ','Financials','USD'),
            ('NEE','Utilities','USD'),
            ('NEM','Materials','USD'),
            ('NFLX','Communication Services','USD'),
            ('NFX','Energy','USD'),
            ('NI','Utilities','USD'),
            ('NKE','Consumer Discretionary','USD'),
            ('NKTR','Health Care','USD'),
            ('NLSN','Industrials','USD'),
            ('NOC','Industrials','USD'),
            ('NOV','Energy','USD'),
            ('NRG','Utilities','USD'),
            ('NSC','Industrials','USD'),
            ('NTAP','Information Technology','USD'),
            ('NTRS','Financials','USD'),
            ('NUE','Materials','USD'),
            ('NVDA','Information Technology','USD'),
            ('NWL','Consumer Discretionary','USD'),
            ('NWS','Communication Services','USD'),
            ('NWSA','Communication Services','USD'),
            ('O','Real Estate','USD'),
            ('OKE','Energy','USD'),
            ('OMC','Communication Services','USD'),
            ('ORCL','Information Technology','USD'),
            ('ORLY','Consumer Discretionary','USD'),
            ('OXY','Energy','USD'),
            ('PAYX','Information Technology','USD'),
            ('PBCT','Financials','USD'),
            ('PCAR','Industrials','USD'),
            ('PEG','Utilities','USD'),
            ('PEP','Consumer Staples','USD'),
            ('PFE','Health Care','USD'),
            ('PFG','Financials','USD'),
            ('PG','Consumer Staples','USD'),
            ('PGR','Financials','USD'),
            ('PH','Industrials','USD'),
            ('PHM','Consumer Discretionary','USD'),
            ('PKG','Materials','USD'),
            ('PKI','Health Care','USD'),
            ('PLD','Real Estate','USD'),
            ('PM','Consumer Staples','USD'),
            ('PNC','Financials','USD'),
            ('PNR','Industrials','USD'),
            ('PNW','Utilities','USD'),
            ('PPG','Materials','USD'),
            ('PPL','Utilities','USD'),
            ('PRGO','Health Care','USD'),
            ('PRU','Financials','USD'),
            ('PSA','Real Estate','USD'),
            ('PSX','Energy','USD'),
            ('PVH','Consumer Discretionary','USD'),
            ('PWR','Industrials','USD'),
            ('PXD','Energy','USD'),
            ('PYPL','Information Technology','USD'),
            ('QCOM','Information Technology','USD'),
            ('QRVO','Information Technology','USD'),
            ('RCL','Consumer Discretionary','USD'),
            ('RE','Financials','USD'),
            ('REG','Real Estate','USD'),
            ('REGN','Health Care','USD'),
            ('RF','Financials','USD'),
            ('RHI','Industrials','USD'),
            ('RHT','Information Technology','USD'),
            ('RJF','Financials','USD'),
            ('RL','Consumer Discretionary','USD'),
            ('RMD','Health Care','USD'),
            ('ROK','Industrials','USD'),
            ('ROL','Industrials','USD'),
            ('ROP','Industrials','USD'),
            ('ROST','Consumer Discretionary','USD'),
            ('RSG','Industrials','USD'),
            ('RTN','Industrials','USD'),
            ('SBAC','Real Estate','USD'),
            ('SBUX','Consumer Discretionary','USD'),
            ('SCHW','Financials','USD'),
            ('SEE','Materials','USD'),
            ('SHW','Materials','USD'),
            ('SIVB','Financials','USD'),
            ('SJM','Consumer Staples','USD'),
            ('SLB','Energy','USD'),
            ('SLG','Real Estate','USD'),
            ('SNA','Industrials','USD'),
            ('SNPS','Information Technology','USD'),
            ('SO','Utilities','USD'),
            ('SPG','Real Estate','USD'),
            ('SPGI','Financials','USD'),
            ('SRE','Utilities','USD'),
            ('STI','Financials','USD'),
            ('STT','Financials','USD'),
            ('STX','Information Technology','USD'),
            ('STZ','Consumer Staples','USD'),
            ('SWK','Industrials','USD'),
            ('SWKS','Information Technology','USD'),
            ('SYF','Financials','USD'),
            ('SYK','Health Care','USD'),
            ('SYMC','Information Technology','USD'),
            ('SYY','Consumer Staples','USD'),
            ('T','Communication Services','USD'),
            ('TAP','Consumer Staples','USD'),
            ('TDG','Industrials','USD'),
            ('TEL','Information Technology','USD'),
            ('TFX','Health Care','USD'),
            ('TGT','Consumer Discretionary','USD'),
            ('TIF','Consumer Discretionary','USD'),
            ('TJX','Consumer Discretionary','USD'),
            ('TMK','Financials','USD'),
            ('TMO','Health Care','USD'),
            ('TPR','Consumer Discretionary','USD'),
            ('TRIP','Communication Services','USD'),
            ('TROW','Financials','USD'),
            ('TRV','Financials','USD'),
            ('TSCO','Consumer Discretionary','USD'),
            ('TSN','Consumer Staples','USD'),
            ('TSS','Information Technology','USD'),
            ('TTWO','Communication Services','USD'),
            ('TWTR','Communication Services','USD'),
            ('TXN','Information Technology','USD'),
            ('TXT','Industrials','USD'),
            ('UA','Consumer Discretionary','USD'),
            ('UAA','Consumer Discretionary','USD'),
            ('UAL','Industrials','USD'),
            ('UDR','Real Estate','USD'),
            ('UHS','Health Care','USD'),
            ('ULTA','Consumer Discretionary','USD'),
            ('UNH','Health Care','USD'),
            ('UNM','Financials','USD'),
            ('UNP','Industrials','USD'),
            ('UPS','Industrials','USD'),
            ('URI','Industrials','USD'),
            ('USB','Financials','USD'),
            ('UTX','Industrials','USD'),
            ('V','Information Technology','USD'),
            ('VAR','Health Care','USD'),
            ('VFC','Consumer Discretionary','USD'),
            ('VIAB','Communication Services','USD'),
            ('VLO','Energy','USD'),
            ('VMC','Materials','USD'),
            ('VNO','Real Estate','USD'),
            ('VRSK','Industrials','USD'),
            ('VRSN','Information Technology','USD'),
            ('VRTX','Health Care','USD'),
            ('VTR','Real Estate','USD'),
            ('VZ','Communication Services','USD'),
            ('WAT','Health Care','USD'),
            ('WBA','Consumer Staples','USD'),
            ('WCG','Health Care','USD'),
            ('WDC','Information Technology','USD'),
            ('WEC','Utilities','USD'),
            ('WELL','Real Estate','USD'),
            ('WFC','Financials','USD'),
            ('WHR','Consumer Discretionary','USD'),
            ('WLTW','Financials','USD'),
            ('WM','Industrials','USD'),
            ('WMB','Energy','USD'),
            ('WMT','Consumer Staples','USD'),
            ('WRK','Materials','USD'),
            ('WU','Information Technology','USD'),
            ('WY','Real Estate','USD'),
            ('WYNN','Consumer Discretionary','USD'),
            ('XEC','Energy','USD'),
            ('XEL','Utilities','USD'),
            ('XLNX','Information Technology','USD'),
            ('XOM','Energy','USD'),
            ('XRAY','Health Care','USD'),
            ('XRX','Information Technology','USD'),
            ('XYL','Industrials','USD'),
            ('YUM','Consumer Discretionary','USD'),
            ('ZBH','Health Care','USD'),
            ('ZION','Financials','USD'),
            ('ZTS','Health Care','USD')
    ]
smi = [
            ('ABBN.SWI','','CHF'),
            ('ADEN.SWI','','CHF'),
            ('BAER.SWI','','CHF'),
            ('CFR.SWI','','CHF'),
            ('CSGN.SWI','','CHF'),
            ('GEBN.SWI','','CHF'),
            ('GIVN.SWI','','CHF'),
            ('LHN.SWI','','CHF'),
            ('LONN.SWI','','CHF'),
            ('NESN.SWI','','CHF'),
            ('NOVN.SWI','','CHF'),
            ('ROG.SWI','','CHF'),
            ('SCMN.SWI','','CHF'),
            ('SGSN.SWI','','CHF'),
            ('SIKA.SWI','','CHF'),
            ('SLHN.SWI','','CHF'),
            ('SREN.SWI','','CHF'),
            ('UBSG.SWI','','CHF'),
            ('UHR.SWI','','CHF'),
            ('ZURN.SWI','','CHF')
    
    ]