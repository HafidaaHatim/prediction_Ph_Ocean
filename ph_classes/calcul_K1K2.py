from ph_classes.CalculConstant import CalculConstant
import math as Math

class calcul_K1K2:
    
    def CalculateRT(self, TempC:float) -> float:
        tempK = TempC + 273.15
        RGasConstant = 83.1451
        return RGasConstant * tempK

    def goyet_poisson(self,salt:float,TempC:float,pres:float,CM:int):
        CC = CalculConstant()
        T = []
        TempK= CC.CalculateTempK(TempC)
        PK1 = 812.27 / TempK + 3.356 - 0.00171 * salt * Math.log(TempK)
        PK1 += 0.000091*salt*salt
        K1 = Math.pow(10, -PK1)
        PK2 = 1450.87 / TempK + 4.604 - 0.00385 * salt * Math.log(TempK)
        PK2 = PK2 + 0.000182 * salt * salt
        K2 = Math.pow(10, -PK2)
        T.append(K1)
        T.append(K2)

        if (pres!=0):
            Pbar = pres / 10
            deltaV1  = -25.5 + 0.1271 * TempC
            Kappa1   = (-3.08 + 0.0877 * TempC) / 1000
            lnK1fac = (-deltaV1 + 0.5 * Kappa1 * Pbar) * Pbar / self.CalculateRT(TempC)
            K1 = K1 * Math.exp(lnK1fac)
            deltaV2  = -15.82 - 0.0219 * TempC
            Kappa2   = (1.13 - 0.1475 * TempC) / 1000
            lnK2fac = (-deltaV2 + 0.5 * Kappa2 * Pbar) * Pbar / self.CalculateRT(TempC)
            K2 = K2 * Math.exp(lnK2fac)
            T.append(K1)
            T.append(K2)
        
        return T

    
    def Roy_et_al(self, salt, TempC,WhoseKSO4,pres, CM):
        CC = CalculConstant()
        T = []
        TempK= CC.CalculateTempK(TempC)

        lnK1 = 2.83655 - 2307.1266 / TempK - 1.5529413 * Math.log(TempK)
        lnK1 = lnK1 + (-0.20760841 - 4.0484 / TempK) * Math.sqrt(salt)
        lnK1 = lnK1 + 0.08468345 * salt - 0.00654208 * Math.sqrt(salt) * salt
        K1 = Math.exp(lnK1)
        K1 = K1 * (1 - 0.001005 * salt)
        K1 = K1 / CC.calculate_SWStoTOT(salt, TempC, WhoseKSO4, pres, CM)
        T.append(K1)

        lnK2 = -9.226508 - 3351.6106 / TempK - 0.2005743 * Math.log(TempK)
        lnK2 = lnK2 + (-0.106901773 - 23.9722 / TempK) * Math.sqrt(salt)
        lnK2 = lnK2 + 0.1130822 * salt - 0.00846934 * Math.sqrt(salt) * salt
        K2 = Math.exp(lnK2)
        K2 = K2 * (1 - 0.001005 * salt)
        K2 = K2 / CC.calculate_SWStoTOT(salt,TempC,WhoseKSO4,pres, CM)
        T.append(K2)

        if pres != 0:
            Pbar = pres / 10
            deltaV1  = -25.5 + 0.1271 * TempC
            Kappa1   = (-3.08 + 0.0877 * TempC) / 1000
            lnK1fac = (-deltaV1 + 0.5 * Kappa1 * Pbar) * Pbar / self.CalculateRT(TempC)
            K1 = K1 * Math.exp(lnK1fac)
            deltaV2  = -15.82 - 0.0219 * TempC
            Kappa2   = (1.13 - 0.1475 * TempC) / 1000
            lnK2fac = (-deltaV2 + 0.5 * Kappa2 * Pbar) * Pbar / self.CalculateRT(TempC)
            K2 = K2 * Math.exp(lnK2fac)
            T.append(K1)
            T.append(K2)

        return T

    
    def MEHRBACH(self, salt, TempC,pres, CM):
        CC = CalculConstant()
        T = []
        TempK= CC.CalculateTempK(TempC)

        pK1 = 3670.7 / TempK - 62.008 + 9.7944 * Math.log(TempK)
        pK1 = pK1 - 0.0118 * salt + 0.000116 * salt * salt
        K1 = Math.pow(10, -pK1)
        T.append(K1)
        pK2 = 1394.7 / TempK + 4.777 - 0.0184 * salt + 0.000118 * salt * salt
        K2 = Math.pow(10, -pK2)
        T.append(K2)

        if pres != 0:
            Pbar = pres / 10
            deltaV1  = -25.5 + 0.1271 * TempC
            Kappa1   = (-3.08 + 0.0877 * TempC) / 1000
            lnK1fac = (-deltaV1 + 0.5 * Kappa1 * Pbar) * Pbar / self.CalculateRT(TempC)
            K1 = K1 * Math.exp(lnK1fac)
            deltaV2  = -15.82 - 0.0219 * TempC
            Kappa2   = (1.13 - 0.1475 * TempC) / 1000
            lnK2fac = (-deltaV2 + 0.5 * Kappa2 * Pbar) * Pbar / self.CalculateRT(TempC)
            K2 = K2 * Math.exp(lnK2fac)
            T.append(K1)
            T.append(K2)

        return T


    def HANSSON_MEHRBACH(self, salt, TempC, pres, CM):
        CC = CalculConstant()
        T = []
        TempK= CC.CalculateTempK(TempC)

        pK1 = 845 / TempK + 3.248 - 0.0098 * salt + 0.000087 * salt * salt
        K1 = Math.pow(10, -pK1)
        T.append(K1)
		
        pK2 = 1377.3 / TempK + 4.824 - 0.0185 * salt + 0.000122 * salt * salt
        K2 = Math.pow(10, -pK2)
        T.append(K2)

        if pres != 0 :
            Pbar = pres / 10
            deltaV1  = -25.5 + 0.1271 * TempC
            Kappa1   = (-3.08 + 0.0877 * TempC) / 1000
            lnK1fac = (-deltaV1 + 0.5 * Kappa1 * Pbar) * Pbar / self.CalculateRT(TempC)
            K1 = K1 * Math.exp(lnK1fac)
            deltaV2  = -15.82 - 0.0219 * TempC
            Kappa2   = (1.13 - 0.1475 * TempC) / 1000
            lnK2fac = (-deltaV2 + 0.5 * Kappa2 * Pbar) * Pbar / self.CalculateRT(TempC)
            K2 = K2 * Math.exp(lnK2fac)
            T.append(K1)
            T.append(K2)

        return T


    def Millero(self, salt, TempC, pres):
        CC = CalculConstant()
        T = []
        TempK= CC.CalculateTempK(TempC)

        lnK1 = 290.9097 - 14554.21 / TempK - 45.0575 * Math.log(TempK)
        K1 = Math.exp(lnK1)
        T.append(K1)
        
        lnK2 = 207.6548 - 11843.79 / TempK - 33.6485 * Math.log(TempK)
        K2 = Math.exp(lnK2)
        T.append(K2)
        if pres != 0 :
            Pbar = pres/10
            deltaV = -30.54 + 0.1849 * TempC - 0.0023366 * TempC * TempC
            Kappa = (-6.22 + 0.1368 * TempC - 0.001233 * TempC * TempC) / 1000
            lnK1fac = (-deltaV + 0.5 * Kappa * Pbar) * Pbar / self.CalculateRT(TempC)
            K1 = K1 * Math.exp(lnK1fac)
            deltaV = -29.81 + 0.115 * TempC - 0.001816 * TempC * TempC
            Kappa = (-5.74 + 0.093 * TempC - 0.001896 * TempC * TempC) / 1000
            lnK2fac = (-deltaV + 0.5 * Kappa * Pbar) * Pbar / self.CalculateRT(TempC)
            K2 = K2 * Math.exp(lnK2fac)
            T.append(K1)
            T.append(K2)

        return T


    def Cai_Wang(self, salt, TempC, pres, CM):
        CC = CalculConstant()
        T = []
        TempK= CC.CalculateTempK(TempC)

        F1 = 200.1 / TempK + 0.322
        pK1 = 3404.71 / TempK + 0.032786 * TempK - 14.8435 - 0.071692 * F1 * Math.pow(salt,0.5) + 0.0021487 * salt
        K1 = Math.pow(10, -pK1)
        K1 = K1 / CC.Calculate_fH(TempC, salt, pres, CM)
        T.append(K1)
	    
        F2 = -129.24 / TempK + 1.4381
        pK2 = 2902.39 / TempK + 0.02379 * TempK - 6.498 - 0.3191 * F2 * Math.pow(salt,0.5) + 0.0198 * salt
        K2 = Math.pow(10, -pK2)
        K2 = K2 / CC.Calculate_fH(TempC, salt, pres, CM)
        T.append(K2)
        if pres != 0:
            Pbar = pres/10
            deltaV = -30.54 + 0.1849 * TempC - 0.0023366 * TempC * TempC
            Kappa = (-6.22 + 0.1368 * TempC - 0.001233 * TempC * TempC) / 1000
            lnK1fac = (-deltaV + 0.5 * Kappa * Pbar) * Pbar / self.CalculateRT(TempC)
            K1 = K1 * Math.exp(lnK1fac)
            
            deltaV = -29.81 + 0.115 * TempC - 0.001816 * TempC * TempC
            Kappa = (-5.74 + 0.093 * TempC - 0.001896 * TempC * TempC) / 1000
            lnK2fac = (-deltaV + 0.5 * Kappa * Pbar) * Pbar / self.CalculateRT(TempC)
            K2 = K2 * Math.exp(lnK2fac)
            T.append(K1)
            T.append(K2)

        return T


    def Mojica_Prieto(self, salt, TempC, pres):
        CC = CalculConstant()
        T = []
        TempK= CC.CalculateTempK(TempC)

        pK1 = -43.6977 - 0.0129037 * salt + 0.0001364 * Math.pow(salt,2) + 2885.378 / TempK + 7.045159 * Math.log(TempK)
        pK2 = -452.094 + 13.142162 * salt - 0.0008101 * Math.pow(salt,2) + 21263.61 / TempK + 68.483143 * Math.log(TempK) 
        + (-581.4428 * salt + 0.259601 * Math.pow(salt,2)) / TempK - 1.967035 * salt * Math.log(TempK)
            
            
        K1 = Math.pow(10, -pK1)
        K2 = Math.pow(10, -pK2)
        T.append(K1)
        T.append(K2)
        if pres != 0 :
            Pbar = pres / 10
            deltaV1  = -25.5 + 0.1271 * TempC
            Kappa1   = (-3.08 + 0.0877 * TempC) / 1000
            lnK1fac = (-deltaV1 + 0.5 * Kappa1 * Pbar) * Pbar / self.CalculateRT(TempC)
            K1 = K1 * Math.exp(lnK1fac)                          
            deltaV2  = -15.82 - 0.0219 * TempC
            Kappa2   = (1.13 - 0.1475 * TempC) / 1000
            lnK2fac = (-deltaV2 + 0.5 * Kappa2 * Pbar) * Pbar / self.CalculateRT(TempC)
            K2 = K2 * Math.exp(lnK2fac)
            T.append(K1)
            T.append(K2)

        return T

    
    def Millero_Al_2006(self, salt, TempC, pres):
        CC = CalculConstant()
        T = []
        TempK= CC.CalculateTempK(TempC)

        pK10 = -126.34048 + 6320.813 / TempK + 19.568224 * Math.log(TempK)
        A1 = 13.4191 * Math.pow(salt,0.5) + 0.0331 * salt - 0.0000533 * Math.pow(salt,2)
        B1 = -530.123 * Math.pow(salt,0.5) - 6.103 * salt
        C1 = -2.0695 * Math.pow(salt,0.5)
        pK1 = A1 + B1 / TempK + C1 * Math.log(TempK) + pK10
        K1 = Math.pow(10, -pK1)
        T.append(K1)
        
        pK20 = -90.18333 + 5143.692 / TempK + 14.613358 * Math.log(TempK)
        A2 = 21.0894 * Math.pow(salt,0.5) + 0.1248 * salt - 0.0003687 * Math.pow(salt,2)	
        B2 = -772.483 * Math.pow(salt,0.5) - 20.051 * salt
        C2 = -3.3336 * Math.pow(salt,0.5)
        pK2 = A2 + B2 / TempK + C2 * Math.log(TempK) + pK20     
        K2 = Math.pow(10, -pK2)
        T.append(K2)
        if pres != 0:
            Pbar = pres / 10
            deltaV1  = -25.5 + 0.1271 * TempC
            Kappa1   = (-3.08 + 0.0877 * TempC) / 1000
            lnK1fac = (-deltaV1 + 0.5 * Kappa1 * Pbar) * Pbar / self.CalculateRT(TempC)
            K1 = K1 * Math.exp(lnK1fac)
            deltaV2  = -15.82 - 0.0219 * TempC
            Kappa2   = (1.13 - 0.1475 * TempC) / 1000
            lnK2fac = (-deltaV2 + 0.5 * Kappa2 * Pbar) * Pbar / self.CalculateRT(TempC)
            K2 = K2 * Math.exp(lnK2fac)
            T.append(K1)
            T.append(K2)

        return T

    
    def Hansson(self, salt, TempC, pres):
        CC = CalculConstant()
        T = []
        TempK= CC.CalculateTempK(TempC)

        pK1 = 851.4 / TempK + 3.237 - 0.0106 * salt + 0.000105 * salt * salt
        K1 = Math.pow(10, -pK1)
        T.append(K1)
		
        pK2 = -3885.4 / TempK + 125.844 - 18.141 * Math.log(TempK)
        pK2 = pK2 - 0.0192 * salt + 0.000132 * salt * salt
        K2 = Math.pow(10, -pK2)
        T.append(K2)
        if pres != 0:
            Pbar = pres / 10
            deltaV1  = -25.5 + 0.1271 * TempC
            Kappa1   = (-3.08 + 0.0877 * TempC) / 1000
            lnK1fac = (-deltaV1 + 0.5 * Kappa1 * Pbar) * Pbar / self.CalculateRT(TempC)
            K1 = K1 * Math.exp(lnK1fac)
            deltaV2  = -15.82 - 0.0219 * TempC
            Kappa2   = (1.13 - 0.1475 * TempC) / 1000
            lnK2fac = (-deltaV2 + 0.5 * Kappa2 * Pbar) * Pbar / self.CalculateRT(TempC)
            K2 = K2 * Math.exp(lnK2fac)
            T.append(K1)
            T.append(K2)

        return T


    def GEOSECS(self, salt, TempC, pres, CM):
        CC = CalculConstant()
        T = []
        TempK= CC.CalculateTempK(TempC)

        logK1 = 13.7201 - 0.031334 * TempK - 3235.76 / TempK
        logK1 = logK1 - 0.000013 * salt * TempK + 0.1032 * Math.sqrt(salt)
        K1 = Math.pow(10, logK1)
        K1 = K1 / CC.Calculate_fH(TempC, salt, pres, CM)
        T.append(K1)
		
        logK2 = -5371.9645 - 1.671221 * TempK + 128375.28 / TempK
        logK2 = logK2 + 2194.3055 * Math.log(TempK) / Math.log(10) - 0.22913 * salt
        logK2 = logK2 - 18.3802 * Math.log(salt) / Math.log(10) + 0.00080944 * salt * TempK
        logK2 = logK2 + 5617.11 * Math.log(salt) / Math.log(10) / TempK - 2.136 * salt / TempK
        K2 = Math.pow(10, logK2)
        K2 = K2 / CC.Calculate_fH(TempC, salt, pres, CM)
        T.append(K2)
        if pres != 0 :
            Pbar=pres/10
            lnK1fac = (24.2 - 0.085 * TempC) * Pbar / self.CalculateRT(TempC)
            lnK2fac = (16.4 - 0.04 * TempC) * Pbar / self.CalculateRT(TempC)
            K1 = K1 * Math.exp(lnK1fac)
            K2 = K2 * Math.exp(lnK2fac)
            T.append(K1)
            T.append(K2)

        return T

    
    def Luecker(self, salt, TempC, WhoseKSO4, pres, CM):
        CC = CalculConstant()
        T = []
        TempK= CC.CalculateTempK(TempC)

        pK1 = 3633.86 / TempK - 61.2172 + 9.6777 * Math.log(TempK) - 0.011555 * salt + 0.0001152 * Math.pow(salt, 2)
        K1 = Math.pow(10, -pK1)
        K1 = K1 / CC.calculate_SWStoTOT(salt, TempC, WhoseKSO4, pres, CM)
        T.append(K1)
        
        pK2 = 471.78 / TempK + 25.929 - 3.16967 * Math.log(TempK) - 0.01781 * salt + 0.0001122 * Math.pow(salt, 2)
        K2 = Math.pow(10, -pK2)
        K2 = K2 / CC.calculate_SWStoTOT(salt, TempC, WhoseKSO4, pres, CM)        
        T.append(K2)
        if pres != 0:
            Pbar = pres / 10
            deltaV1  = -25.5 + 0.1271 * TempC
            Kappa1   = (-3.08 + 0.0877 * TempC) / 1000
            lnK1fac = (-deltaV1 + 0.5 * Kappa1 * Pbar) * Pbar / self.CalculateRT(TempC)
            K1 = K1 * Math.exp(lnK1fac)
            deltaV2  = -15.82 - 0.0219 * TempC
            Kappa2   = (1.13 - 0.1475 * TempC) / 1000
            lnK2fac = (-deltaV2 + 0.5 * Kappa2 * Pbar) * Pbar / self.CalculateRT(TempC)
            K2 = K2 * Math.exp(lnK2fac)
            T.append(K1)
            T.append(K2)

        return T


    def Millero_al_2002(self, salt, TempC, pres):
        CC = CalculConstant()
        T = []
        TempK = CC.CalculateTempK(TempC)

        pK1 = 6.359 - 0.00664 * salt - 0.01322 * TempC + 0.00004989 * Math.pow(TempC, 2)
        pK2 = 9.867 - 0.01314 * salt - 0.01904 * TempC + 0.00002448 * Math.pow(TempC, 2)
        K1 = Math.pow(10, -pK1)
        K2 = Math.pow(10, -pK2)
        
        T.append(K1)
        T.append(K2)
        if pres != 0:
            Pbar = pres / 10
            deltaV1  = -25.5 + 0.1271 * TempC
            Kappa1   = (-3.08 + 0.0877 * TempC) / 1000
            lnK1fac = (-deltaV1 + 0.5 * Kappa1 * Pbar) * Pbar / self.CalculateRT(TempC)
            K1 = K1 * Math.exp(lnK1fac)
            deltaV2  = -15.82 - 0.0219 * TempC
            Kappa2   = (1.13 - 0.1475 * TempC) / 1000
            lnK2fac = (-deltaV2 + 0.5 * Kappa2 * Pbar) * Pbar / self.CalculateRT(TempC)
            K2 = K2 * Math.exp(lnK2fac)
            T.append(K1)
            T.append(K2)

        return T


    def Millero_2010(self, salt, TempC, pres):
        CC = CalculConstant()
        T = []
        TempK= CC.CalculateTempK(TempC)

        pK10 = -126.34048 + 6320.813 / TempK + 19.568224 * Math.log(TempK)
        A1 = 13.4038 * Math.pow(salt, 0.5) + 0.03206 * salt - 0.00005242 * Math.pow(salt, 2)
        B1 = -530.659 * Math.pow(salt, 0.5) - 5.821 * salt
        C1 = -2.0664 * Math.pow(salt, 0.5)
        pK1 = pK10 + A1 + B1 / TempK + C1 * Math.log(TempK)
        K1 = Math.pow(10, -pK1)
        T.append(K1)
      
        pK20 = -90.18333 + 5143.692 / TempK + 14.613358 * Math.log(TempK)
        A2 = 21.3728 * Math.pow(salt, 0.5) + 0.1218 * salt - 0.0003688 * Math.pow(salt, 2)
        b2 = -788.289 * Math.pow(salt, 0.5) - 19.189 * salt
        C2 = -3.374 * Math.pow(salt, 0.5)
        pK2 = pK20 + A2 + b2 / TempK + C2 * Math.log(TempK)
        K2 = Math.pow(10, -pK2)
        T.append(K2)
        if pres != 0:
            Pbar = pres / 10
            deltaV1  = -25.5 + 0.1271 * TempC
            Kappa1   = (-3.08 + 0.0877 * TempC) / 1000
            lnK1fac = (-deltaV1 + 0.5 * Kappa1 * Pbar) * Pbar / self.CalculateRT(TempC)
            K1 = K1 * Math.exp(lnK1fac)
            deltaV2  = -15.82 - 0.0219 * TempC
            Kappa2   = (1.13 - 0.1475 * TempC) / 1000
            lnK2fac = (-deltaV2 + 0.5 * Kappa2 * Pbar) * Pbar / self.CalculateRT(TempC)
            K2 = K2 * Math.exp(lnK2fac)
            T.append(K1)
            T.append(K2)

        return T