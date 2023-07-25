import math as m
class CalculConstant:
    
    # calculate TempK
    def CalculateTempK(self,TempC:float):
        TempK = TempC + 273.15
        return TempK

    # calculate RT
    def CalculateRT(self, TempC:float):
        RGasConstant = 83.1451
        RT = RGasConstant * self.CalculateTempK(TempC)
        return RT

    # calculate fH
    def Calculate_fH(self, TempC:float, Salinity:float, pres:float, CM:int):
        
        if CM == 7:
            fH = 1.29 - 0.00204 * self.CalculateTempK(TempC)
            fH = fH + (0.00046 - 0.00000148 * self.CalculateTempK(TempC)) * Salinity * Salinity
        elif CM == 8:
            fH = 1
        else:
            fH = 1.2948 - 0.002036 * self.CalculateTempK(TempC)
            fH = fH + (0.0004607 - 0.000001475 * self.CalculateTempK(TempC)) * Salinity * Salinity
        
        return fH

    # calculate KB
    def calculate_KB(self, salt: float, TempC: float, pres: float, cm: int, WhoseKSO4: int, pHscale: int) -> float:
        temp_k = self.CalculateTempK(TempC)
        kb = 0

        if cm == 7:
            log_kb = -9.26 + 0.00886 * salt + 0.01 * TempC
            kb = pow(10, log_kb) / self.Calculate_fH(TempC, salt, pres, cm)
        elif cm == 8:
            kb = 0
        else:
            ln_kb_top = -8966.9 - 2890.53 * m.sqrt(salt) - 77.942 * salt
            ln_kb_top += 1.728 * m.sqrt(salt) * salt - 0.0996 * pow(salt, 2)
            ln_kb = ln_kb_top / temp_k
            ln_kb += 148.0248 + 137.1942 * m.sqrt(salt) + 1.62142 * salt
            ln_kb += (-24.4344 - 25.085 * m.sqrt(salt) - 0.2474 * salt) * m.log(temp_k)
            ln_kb += 0.053105 * m.sqrt(salt) * temp_k
            kb = m.exp(ln_kb)
            kb = kb / self.calculate_SWStoTOT(salt, TempC, WhoseKSO4, pres, cm)
        
        if pres != 0:
            Pbar = pres/10
            if cm in (6, 7):
                ln_kb_fac = (27.5 - 0.095 * TempC) * Pbar / self.CalculateRT(TempC)
                kb = kb * m.exp(ln_kb_fac)
            elif cm == 8:
                ln_kb_fac = 0                
                kb = kb * m.exp(ln_kb_fac)
            else:
                deltaV = -29.48 + 0.1622 * TempC - 0.002608 * m.pow(TempC, 2)
                Kappa = -2.84 / 1000
                lnKBfac = (-deltaV + 0.5 * Kappa * Pbar) * Pbar / self.CalculateRT(TempC)
                KBfac  = m.exp(lnKBfac)
                kb = kb * KBfac
        
        if pHscale == 1 : kb = kb * self.calculate_SWStoTOT(salt, TempC, WhoseKSO4, pres, cm)
        elif pHscale == 3 : kb = kb * self.calculate_SWStoTOT(salt,TempC,WhoseKSO4,pres,cm) / self.calculate_FREEtoTOT(salt,TempC,WhoseKSO4,pres,cm)
        elif pHscale == 4 : kb * self.Calculate_fH(TempC,salt,pres,cm)

        return kb

    
    # Calculate KF
    def CalculateKF(self, salinity:float, TempC:float, pres:float, CM:int) -> float:
        TempK = self.CalculateTempK(TempC)
        IonS = self.CalculateIonS(salinity, pres, CM)
        lnKF = 1590.2 / TempK - 12.641 + 1.525 * m.sqrt(IonS)
        KF = m.exp(lnKF)
        KF = KF * (1 - 0.001005 * salinity)

        if pres != 0:
            Pbar = pres / 10
            deltaV = -9.78 - 0.009 * TempC - 0.000942 * TempC * TempC
            Kappa = (-3.91 + 0.054 * TempC) / 1000
            lnKFFac = (-deltaV + 0.5 * Kappa * Pbar) * Pbar / self.CalculateRT(TempC)
            KF = KF * m.exp(lnKFFac)
        
        return KF

    
    # Calculate TB
    def CalculateTB(self, WhichTB:int, salinity:float, pres:float, CM:int) -> float:
        TB = 0.0
        if CM in (6, 7): TB = 0.0004106 * salinity / 35
        elif CM == 8 : TB = 0
        else:
            if WhichTB == 1 : TB = (0.0004157 * salinity) / 35
            else : TB = (0.0004326 * salinity) / 35
        
        return TB

    # Calculate TS
    def CalculateTS(self, salinity:float, pres:float,CM:int) -> float:
        return (0.14 / 96.062) * (salinity / 1.80655)
        

    # Calculate TF
    def CalculateTF(self, Salinity:float,pres:float,CM:int) -> float:
        return (0.000067 / 18.998) * (Salinity / 1.80655)
        
    # calculate KS
    def CalculateKS(self, WhoseKS04:int, salinity:float,TempC:float,pres:float,CM:int) -> float:
        TempK = TempC + 273.15
        IonS = self.CalculateIonS(salinity, pres, CM)
        KS = 0.0

        if WhoseKS04 == 1:
            lnKS = -4276.1 / TempK + 141.328 - 23.093 * m.log(TempK)
            lnKS = lnKS + (-13856 / TempK + 324.57 - 47.986 * m.log(TempK)) * m.sqrt(IonS)
            lnKS = lnKS + (35474 / TempK - 771.54 + 114.723 * m.log(TempK)) * IonS
            lnKS = lnKS + (-2698 / TempK) * m.sqrt(IonS) * IonS
            lnKS = lnKS + (1776 / TempK) * IonS * IonS
            KS = m.exp(lnKS)
            KS = KS * (1 - 0.001005 * salinity)
        elif WhoseKS04 == 2:
            pKS = 647.59 / TempK - 6.3451 + 0.019085 * TempK - 0.5208 * m.sqrt(IonS)
            KS = m.pow(10, pKS)
            KS = KS * (1 - 0.001005 * salinity)
        
        if pres != 0:
            Pbar = pres / 10
            deltaV = -18.03 + 0.0466 * TempC + 0.000316 * TempC * TempC
            Kappa = (-4.53 + 0.09 * TempC) / 1000
            lnKSFac = (-deltaV + 0.5 * Kappa * Pbar) * Pbar / self.CalculateRT(TempC)
            KS = KS * m.exp(lnKSFac)

        return KS

    def CalculateIonS(self, salinity:float,pres:float,CM:int) -> float:
        return 19.924 * salinity / (1000 - 1.005 * salinity)

    
    # calculate KW
    def CalculateKW(self,Salinity:float,TempC:float,pres:float,CM:int,WhoseKSO4:int,pHscale:int) -> float:
        KW = 0
        TempK = TempC + 273.15
        if CM == 6: KW = 0
        elif CM == 7:
            lnKW = 148.9802 - 13847.26 / TempK - 23.6521 * m.log(TempK)
            lnKW = lnKW + (-79.2447 + 3298.72 / TempK + 12.0408 * m.log(TempK)) * m.sqrt(Salinity)
            lnKW = lnKW - 0.019813 * Salinity
            KW = m.exp(lnKW)
        elif CM == 8:
            lnKW = 148.9802 - 13847.26 / TempK - 23.6521 * m.log(TempK)
            KW = m.exp(lnKW)
        else:
            lnKW = 148.9802 - 13847.26 / TempK - 23.6521 * m.log(TempK)
            lnKW = lnKW + (-5.977 + 118.67 / TempK + 1.0495 * m.log(TempK)) * m.sqrt(Salinity)
            lnKW = lnKW - 0.01615 * Salinity
            KW = m.exp(lnKW)
        
        if pres != 0:
            if CM == 8:
                Pbar = pres / 10
                deltaV = -25.6 + 0.2324 * TempC - 0.0036246 * TempC * TempC
                Kappa = (-7.33 + 0.1368 * TempC - 0.001233 * TempC * TempC) / 1000
                lnKWFac  = (-deltaV + 0.5 * Kappa * Pbar) * Pbar / self.CalculateRT(TempC)
                KW = KW * m.exp(lnKWFac)
            else:
                Pbar = pres / 10
                deltaV = -20.02 + 0.1119 * TempC - 0.001409 * TempC * TempC
                Kappa = (-5.13 + 0.0794 * TempC) / 1000
                lnKWFac = (-deltaV + 0.5 * Kappa * Pbar) * Pbar / self.CalculateRT(TempC)
                KW= KW * m.exp(lnKWFac)
        
        if pHscale == 1 : KW = KW * self.calculate_SWStoTOT(Salinity,TempC,WhoseKSO4,pres,CM)
        elif pHscale == 3 : KW = KW * self.calculate_SWStoTOT(Salinity,TempC,WhoseKSO4,pres,CM) / self.calculate_FREEtoTOT(Salinity,TempC,WhoseKSO4,pres,CM)
        elif pHscale == 4 : KW = KW * self.Calculate_fH(TempC,Salinity,pres,CM)

        return KW

    # calculate KP1
    def CalculateKP1(self,salinity:float,TempC:float,pres:float,CM:int,WhoseKSO4:int,pHscale:int) -> float:
        TempK = TempC+ 273.15

        if CM == 7 : KP1 = 0.02
        elif CM == 8 : KP1 = 0
        else : 
            lnKP1 = -4576.752 / TempK + 115.54 - 18.453 * m.log(TempK)
            lnKP1 = lnKP1 + (-106.736 / TempK + 0.69171) * m.sqrt(salinity)
            lnKP1 = lnKP1 + (-0.65643 / TempK - 0.01844) * salinity
            KP1 = m.exp(lnKP1)
        
        if pres != 0 :
            Pbar = pres / 10
            deltaV = -14.51 + 0.1211 * TempC - 0.000321 * TempC * TempC
            Kappa = (-2.67 + 0.0427 * TempC) / 1000
            lnKP1Fac = (-deltaV + 0.5 * Kappa * Pbar) * Pbar / self.CalculateRT(TempC)
            KP1 = KP1 * m.exp(lnKP1Fac)

        if pHscale == 1 : KP1 = KP1 * self.calculate_SWStoTOT(salinity,TempC,WhoseKSO4,pres,CM)
        elif pHscale == 3 : KP1 = KP1 * self.calculate_SWStoTOT(salinity,TempC,WhoseKSO4,pres,CM) / self.calculate_FREEtoTOT(salinity,TempC,WhoseKSO4,pres,CM)
        elif pHscale == 4 : KP1 = KP1 * self.Calculate_fH(TempC,salinity,pres,CM)

        return KP1

    
    # calculate KP2
    def CalculateKP2(self,salinity:float,TempC:float,pres:float,CM:int,WhoseKSO4:int,pHscale:int) -> float:
        TempK = TempC + 273.15

        if CM == 7 : 
            KP2 = m.exp(-9.039 - 1450 / TempK)
            KP2 = KP2 / self.Calculate_fH(TempC, salinity, pres, CM)
        elif CM == 8 : KP2 = 0
        else : 
            lnKP2 = -8814.715 / TempK + 172.1033 - 27.927 * m.log(TempK)
            lnKP2 = lnKP2 + (-160.34 / TempK + 1.3566) * m.sqrt(salinity)
            lnKP2 = lnKP2 + (0.37335 / TempK - 0.05778) * salinity
            KP2 = m.exp(lnKP2)
        
        if pres != 0 :
            Pbar = pres / 10
            deltaV = -23.12 + 0.1758 * TempC - 0.002647 * TempC * TempC
            Kappa = (-5.15 + 0.09 * TempC) / 1000
            lnKP2Fac = (-deltaV + 0.5 * Kappa * Pbar) * Pbar / self.CalculateRT(TempC)
            KP2 = KP2 * m.exp(lnKP2Fac)

        if pHscale == 1 : KP2 = KP2 * self.calculate_SWStoTOT(salinity,TempC,WhoseKSO4,pres,CM)
        elif pHscale == 3 : KP2 = KP2 * self.calculate_SWStoTOT(salinity,TempC,WhoseKSO4,pres,CM) / self.calculate_FREEtoTOT(salinity,TempC,WhoseKSO4,pres,CM)
        elif pHscale == 4 : KP2 = KP2 * self.Calculate_fH(TempC,salinity,pres,CM)

        return KP2

    # calculate KP3
    def CalculateKP3(self,salinity:float,TempC:float,pres:float,CM:int,WhoseKSO4:int,pHscale:int) -> float:
        TempK = TempC + 273.15

        if CM == 7 : 
            KP3 = m.exp(4.466 - 7276 / TempK)
            KP3 = KP3 / self.Calculate_fH(TempC, salinity, pres, CM)
        elif CM == 8 : KP3 = 0
        else : 
            lnKP3 = -3070.75 / TempK - 18.126
            lnKP3 = lnKP3 + (17.27039 / TempK + 2.81197) * m.sqrt(salinity)
            lnKP3 = lnKP3 + (-44.99486 / TempK - 0.09984) * salinity
            KP3 = m.exp(lnKP3)
        
        if pres != 0 :
            Pbar = pres / 10
            deltaV = -26.57 + 0.202 * TempC - 0.003042 * TempC * TempC
            Kappa = (-4.08 + 0.0714 * TempC) / 1000
            lnKP3Fac = (-deltaV + 0.5 * Kappa * Pbar) * Pbar / self.CalculateRT(TempC)
            KP3 = KP3 * m.exp(lnKP3Fac)

        if pHscale == 1 : KP3 = KP3 * self.calculate_SWStoTOT(salinity,TempC,WhoseKSO4,pres,CM)
        elif pHscale == 3 : KP3 = KP3 * self.calculate_SWStoTOT(salinity,TempC,WhoseKSO4,pres,CM) / self.calculate_FREEtoTOT(salinity,TempC,WhoseKSO4,pres,CM)
        elif pHscale == 4 : KP3 = KP3 * self.Calculate_fH(TempC,salinity,pres,CM)

        return KP3

    # calculate KSi
    def CalculateKSi(self,salinity:float,TempC:float,pres:float,CM:int,WhoseKSO4:int,pHscale:int) -> float:
        TempK = TempC+ 273.15
        IonS = self.CalculateIonS(salinity, pres, CM)

        if CM == 8 : KSi = 0
        elif CM == 7 :
            KSi = 0.0000000004
            KSi = KSi / self.Calculate_fH(TempC, salinity, pres, CM)
        else:
            lnKSi = -8904.2 / TempK + 117.4 - 19.334 * m.log(TempK)
            lnKSi = lnKSi + (-458.79 / TempK + 3.5913) * m.sqrt(IonS)
            lnKSi = lnKSi + (188.74 / TempK - 1.5998) * IonS
            lnKSi = lnKSi + (-12.1652 / TempK + 0.07871) * IonS * IonS
            KSi = m.exp(lnKSi)
            KSi = KSi * (1 - 0.001005 * salinity)

        if pres != 0:
            Pbar = pres / 10
            deltaV = -29.48 + 0.1622 * TempC - 0.002608 * TempC * TempC
            Kappa = -2.84 / 1000
            lnKsiFac = (-deltaV + 0.5 * Kappa * Pbar) * Pbar / self.CalculateRT(TempC)
            KSi = KSi * m.exp(lnKsiFac)

        if pHscale == 1 : KSi = KSi * self.calculate_SWStoTOT(salinity,TempC,WhoseKSO4,pres,CM)
        elif pHscale == 3 : KSi = KSi * self.calculate_SWStoTOT(salinity,TempC,WhoseKSO4,pres,CM) / self.calculate_FREEtoTOT(salinity,TempC,WhoseKSO4,pres,CM)
        elif pHscale == 4 : KSi = KSi * self.Calculate_fH(TempC,salinity,pres,CM)

        return KSi

    
    # calculate K0
    def CalculateK0(self,salinity:float,TempC:float,pres:float,CM:int) -> float:
        lnK0,TIK = 0.0, 0.0
        TempK = self.CalculateTempK(TempC)

        TIK = TempK / 100
        lnK0 = -60.2409 + 93.4517 / TIK + 23.3585 * m.log(TIK)
        lnK0 = lnK0 + salinity * (0.023517 - 0.023656 * TIK + 0.0047036 * TIK * TIK)
        K0 = m.exp(lnK0)

        return K0


    # calculate SWStoTOT
    def calculate_SWStoTOT(self, salinity:float, TempC:float, WhoseKSO4:int, pres:float,CM:int) -> float:
        SWStoTOT = 0.0
        ts = self.CalculateTS(salinity, pres, CM)
        ks = self.CalculateKS(WhoseKSO4, salinity, TempC, pres, CM)
        tf = self.CalculateTF(salinity, pres, CM)
        kf = self.CalculateKF(salinity, TempC, pres, CM)
        SWStoTOT = (1 + ts / ks) / (1 + ts / ks + tf / kf)
        return SWStoTOT

    # calculate FREEtoTOT
    def calculate_FREEtoTOT(self,salinity:float,TempC:float,WhoseKSO4:int,pres:float,CM:int) -> float:
        FREEtoTOT=0.0
        TS = self.CalculateTS(salinity, pres, CM)
        KS = self.CalculateKS(WhoseKSO4, salinity, TempC, pres, CM)
        FREEtoTOT = 1 + TS / KS

        return FREEtoTOT

    # calculate Fugacity
    def CalculateFugacityConstants(self,TempC:float) -> float:
        TempK = self.CalculateTempK(TempC)
        RT = self.CalculateRT(TempC)
        FugFac = 0.0
        
        Delta = (57.7 - 0.118 * TempK)
        b = (-1636.75 + 12.0408 * TempK - 0.0327957 * m.pow(TempK, 2) + 3.16528 * 0.00001 * m.pow(TempK, 3))
        P1atm = 1.01325
        
        FugFac = m.exp((b + 2 * Delta) * P1atm /RT)
            
        return FugFac

    