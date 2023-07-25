from ph_classes.CalculConstant import CalculConstant

class GenerateConstantes:

    Ca = CalculConstant()

    def remplirCdict(self,Salinity:float,WhichTB:int,TempC:float,WhoseKSO4:int,pres:float,CM:int,pHscale:int):

        cdict = [ 
            ["KB", "TB", "TS", "TF", "KF", "KS", "KW", "KP1", "KP2", "KP3", "KSi", "K0", "K1", "K2", "FugFac", "SWStoTOT", "FREEtoTOT", "fH"],
            [
                str(self.Ca.calculate_KB(Salinity, TempC, pres, CM,WhoseKSO4,pHscale)),
                str(self.Ca.CalculateTB(WhichTB, Salinity,pres,CM)),
                str(self.Ca.CalculateTS(Salinity,pres,CM)),
                str(self.Ca.CalculateTF(Salinity,pres,CM)),
                str(self.Ca.CalculateKF(Salinity, TempC,pres,CM)),
                str(self.Ca.CalculateKS(WhoseKSO4, Salinity, TempC,pres,CM)),
                str(self.Ca.CalculateKW(Salinity, TempC,pres,CM,WhoseKSO4,pHscale)),
                str(self.Ca.CalculateKP1(Salinity, TempC,pres,CM,WhoseKSO4,pHscale)),
                str(self.Ca.CalculateKP2(Salinity, TempC,pres,CM,WhoseKSO4,pHscale)),
                str(self.Ca.CalculateKP3(Salinity, TempC,pres,CM ,WhoseKSO4,pHscale)),
                str(self.Ca.CalculateKSi(Salinity, TempC,pres,CM,WhoseKSO4,pHscale)),
                str(self.Ca.CalculateK0(Salinity, TempC,pres,CM)),
                str(0),
                str(0),
                str(self.Ca.CalculateFugacityConstants(TempC)),
                str(self.Ca.calculate_SWStoTOT(Salinity, TempC, WhoseKSO4,pres,CM)),
                str(self.Ca.calculate_FREEtoTOT(Salinity, TempC, WhoseKSO4,pres,CM)),
                str(self.Ca.Calculate_fH(TempC, Salinity,pres,CM))
            ]            
        ]

        return cdict

    def display(self, matrix):
        for j in range(len(matrix[0])):
            print(matrix[0][j], " :\t", matrix[0][j], "\n")