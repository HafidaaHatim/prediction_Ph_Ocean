import math as Math
from ph_classes.RemplirTableK1K2 import RemplirTableK1K2

def CalculpHfromTATC(TC, TA, Salinity, WhichTB, TempC, WhoseKSO4, pres, TP, TSi, CM, ph_scale):
    Tab = RemplirTableK1K2()
    T= Tab.RemplirTableK1K2WithPress(Salinity, WhichTB, TempC, WhoseKSO4, pres,CM,ph_scale)
    pHGuess = 8
    pHTol = 0.0001
    ln10 = Math.log(10)
    pH = pHGuess
    i=0

    H = Math.pow(10, -pH)
    Denom = (H*H + float(T[1][12])*H + float(T[1][12])*float(T[1][13]))
    CAlk = TC*Math.pow(10, -6)*float(T[1][12]) * (H + 2*float(T[1][13])) / Denom
    BAlk = float(T[1][1]) * float(T[1][0]) / (float(T[1][0]) + H)
    OH   = float(T[1][6]) / H
    PhosTop = float(T[1][7]) * float(T[1][8]) * H + 2 * float(T[1][7]) * float(T[1][8]) * float(T[1][9]) - H * H * H
    PhosBot = H * H * H + float(T[1][7]) * H * H + float(T[1][7]) * float(T[1][8]) * H + float(T[1][7]) * float(T[1][8]) * float(T[1][9])
    PAlk = TP*Math.pow(10, -6) * PhosTop / PhosBot
    SiAlk = TSi*Math.pow(10, -6)*float(T[1][10]) / (float(T[1][10]) + H)
    Hfree = H / float(T[1][16])
    HSO4  = float(T[1][2]) / (1 + float(T[1][5])/Hfree)
    HF = float(T[1][3]) / (1 + float(T[1][4])/Hfree)
    Residual  = TA*Math.pow(10, -6) - CAlk - BAlk - OH - PAlk - SiAlk + Hfree + HSO4 + HF;        
    Slope = ln10 * (TC*Math.pow(10, -6)*float(T[1][12])*H * (H*H + float(T[1][12])*float(T[1][13]) + 4*H*float(T[1][13])) / Denom / Denom + BAlk*H / (float(T[1][0]) + H) + OH + H)
    deltapH   = Residual / Slope
    deltapH = deltapH / 2

    while abs(deltapH) > 1:
        deltapH = deltapH / 2
        
    pH = pH + deltapH

    while abs(deltapH) > pHTol:
        H = Math.pow(10, -pH)
        Denom = (H*H + float(T[1][12])*H + float(T[1][12])*float(T[1][13]))
        CAlk = TC*Math.pow(10, -6)*float(T[1][12]) * (H + 2*float(T[1][13])) / Denom
        BAlk = float(T[1][1]) * float(T[1][0]) / (float(T[1][0]) + H)
        OH   = float(T[1][6]) / H
        PhosTop = float(T[1][7]) * float(T[1][8]) * H + 2 * float(T[1][7]) * float(T[1][8]) * float(T[1][9]) - H * H * H
        PhosBot = H * H * H + float(T[1][7]) * H * H + float(T[1][7]) * float(T[1][8]) * H + float(T[1][7]) * float(T[1][8]) * float(T[1][9])
        PAlk = TP*Math.pow(10, -6) * PhosTop / PhosBot
        SiAlk = TSi*Math.pow(10, -6)*float(T[1][10]) / (float(T[1][10]) + H)
        Hfree = H / float(T[1][16])
        HSO4  = float(T[1][2]) / (1 + float(T[1][5])/Hfree)
        HF = float(T[1][3]) / (1 + float(T[1][4])/Hfree)
        Residual  = TA*Math.pow(10, -6) - CAlk - BAlk - OH - PAlk - SiAlk + Hfree + HSO4 + HF;        
        Slope =ln10 * (TC*Math.pow(10, -6)*float(T[1][12])*H * (H*H + float(T[1][12])*float(T[1][13]) + 4*H*float(T[1][13])) /Denom/Denom + BAlk*H / (float(T[1][0]) + H) + OH + H)
        deltapH   = Residual / Slope
        deltapH = deltapH / 2

        while abs(deltapH) > 1:
            deltapH = deltapH / 2
        
        pH = pH + deltapH

    return pH