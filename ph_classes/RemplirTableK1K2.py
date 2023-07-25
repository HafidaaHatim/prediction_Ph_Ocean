from ph_classes.GenerateConstantes import GenerateConstantes
from ph_classes.CalculConstant import CalculConstant
from ph_classes.calcul_K1K2 import calcul_K1K2

class RemplirTableK1K2:
    Ge = GenerateConstantes()
    Cn = CalculConstant()
    cK = calcul_K1K2()

    def RemplirTableK1K2WithPress(self, Salinity, WhichTB, TempC, WhoseKSO4, pres, CM, pHscale):
        T =  self.Ge.remplirCdict(Salinity, WhichTB,  TempC,  WhoseKSO4, pres, CM,pHscale)
        
        if CM == 1: K = self.cK.goyet_poisson(Salinity,TempC,pres, CM)
        elif CM == 2: K = self.cK.Roy_et_al(Salinity,TempC, WhoseKSO4, pres, CM)
        elif CM == 3: K = self.cK.Hansson(Salinity, TempC,pres)
        elif CM == 4: K = self.cK.MEHRBACH(Salinity, TempC, pres, CM)
        elif CM == 5: K = self.cK.HANSSON_MEHRBACH(Salinity, TempC, pres, CM)
        elif CM == 6: K = self.cK.GEOSECS(Salinity, TempC, pres, CM)
        elif CM == 7: K = self.cK.Millero(Salinity, TempC, pres)
        elif CM == 8: K = self.cK.Cai_Wang(Salinity, TempC, pres, CM)
        elif CM == 9: K = self.cK.Luecker(Salinity, TempC, WhoseKSO4, pres, CM)
        elif CM == 10: K = self.cK.Mojica_Prieto(Salinity, TempC,pres)
        elif CM == 11: K = self.cK.Millero_al_2002(Salinity, TempC, pres)
        elif CM == 12: K = self.cK.Millero_Al_2006(Salinity, TempC, pres)
        elif CM == 13: K = self.cK.Millero_2010(Salinity, TempC,pres)

        if pHscale == 1:
            K[0] = K[0] * self.Cn.calculate_SWStoTOT(Salinity, TempC, WhoseKSO4, pres, CM)
            K[1] = K[1] * self.Cn.calculate_SWStoTOT(Salinity, TempC, WhoseKSO4, pres, CM)
            T[1][6] = str(float(T[1][6])*self.Cn.calculate_SWStoTOT(Salinity, TempC, WhoseKSO4, pres, CM))
            T[1][0] = str(float(T[1][0])*self.Cn.calculate_SWStoTOT(Salinity, TempC, WhoseKSO4, pres, CM))
            T[1][7] = str(float(T[1][7])*self.Cn.calculate_SWStoTOT(Salinity, TempC, WhoseKSO4, pres, CM))
            T[1][8] = str(float(T[1][8])*self.Cn.calculate_SWStoTOT(Salinity, TempC, WhoseKSO4, pres, CM))
            T[1][9] = str(float(T[1][9])*self.Cn.calculate_SWStoTOT(Salinity, TempC, WhoseKSO4, pres, CM))
            T[1][10] = str(float(T[1][10])*self.Cn.calculate_SWStoTOT(Salinity, TempC, WhoseKSO4, pres, CM))
        elif pHscale == 3:
            K[0] = K[0] * self.Cn.calculate_SWStoTOT(Salinity, TempC, WhoseKSO4, pres, CM) / self.Cn.calculate_FREEtoTOT(Salinity, TempC, WhoseKSO4, pres, CM)
            K[1] = K[1] * self.Cn.calculate_SWStoTOT(Salinity, TempC, WhoseKSO4, pres, CM) / self.Cn.calculate_FREEtoTOT(Salinity, TempC, WhoseKSO4, pres, CM)
            T[1][6] = str(float(T[1][6])*self.Cn.calculate_SWStoTOT(Salinity, TempC, WhoseKSO4, pres, CM) / self.Cn.calculate_FREEtoTOT(Salinity, TempC, WhoseKSO4, pres, CM))
            T[1][0] = str(float(T[1][0])*self.Cn.calculate_SWStoTOT(Salinity, TempC, WhoseKSO4, pres, CM) / self.Cn.calculate_FREEtoTOT(Salinity, TempC, WhoseKSO4, pres, CM))
            T[1][7] = str(float(T[1][7])*self.Cn.calculate_SWStoTOT(Salinity, TempC, WhoseKSO4, pres, CM) / self.Cn.calculate_FREEtoTOT(Salinity, TempC, WhoseKSO4, pres, CM))
            T[1][8] = str(float(T[1][8])*self.Cn.calculate_SWStoTOT(Salinity, TempC, WhoseKSO4, pres, CM) / self.Cn.calculate_FREEtoTOT(Salinity, TempC, WhoseKSO4, pres, CM))
            T[1][9] = str(float(T[1][9])*self.Cn.calculate_SWStoTOT(Salinity, TempC, WhoseKSO4, pres, CM) / self.Cn.calculate_FREEtoTOT(Salinity, TempC, WhoseKSO4, pres, CM))
            T[1][10] = str(float(T[1][10])*self.Cn.calculate_SWStoTOT(Salinity, TempC, WhoseKSO4, pres, CM) / self.Cn.calculate_FREEtoTOT(Salinity, TempC, WhoseKSO4, pres, CM))
        elif pHscale == 4:
            K[0] = K[0] * self.Cn.Calculate_fH(TempC, Salinity, pres, CM)
            K[1] = K[1] * self.Cn.Calculate_fH(TempC, Salinity, pres, CM)
            T[1][6] = str(float(T[1][6])*self.Cn.Calculate_fH(TempC, Salinity, pres, CM))
            T[1][0] = str(float(T[1][0])*self.Cn.Calculate_fH(TempC, Salinity, pres, CM))
            T[1][7] = str(float(T[1][7])*self.Cn.Calculate_fH(TempC, Salinity, pres, CM))
            T[1][8] = str(float(T[1][8])*self.Cn.Calculate_fH(TempC, Salinity, pres, CM))
            T[1][9] = str(float(T[1][9])*self.Cn.Calculate_fH(TempC, Salinity, pres, CM))
            T[1][10] = str(float(T[1][10])*self.Cn.Calculate_fH(TempC, Salinity, pres, CM))

        T[1][12]= str(K[0])
        T[1][13]= str(K[1])

        return T