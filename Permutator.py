import sys


class DigitCounter():

    def __init__(self,NDigits, MaxDigits=None):
        self.NDigits=NDigits
        if MaxDigits:
            self.MaxDigits=MaxDigits
        else:
            self.MaxDigits=[9]*NDigits
            
        self.Digits=[0]*NDigits
        self.Done=False

    def Increment(self):
        if self.Done:
            return False
        
        for i in xrange(self.NDigits-1,-1,-1):
            if self.Digits[i]>=self.MaxDigits[i]:
                if i==0:
                    self.Done=True
                self.Digits[i]=0
            else:
                self.Digits[i]+=1
                break
        return True

    def Reset(self):
        self.Digits=[0]*self.NDigits
        self.Done=False
            
class Permutator():
    def __init__(self,Params):
        self.Params=Params
        DigitMaxs=[0]*len(Params.keys())

        i=0
        self.keys=Params.keys()
        self.keys.sort()
        
        for p in self.keys:
            DigitMaxs[i]=len(Params[p])-1
            i+=1

        self.Digits=DigitCounter(len(Params.keys()),DigitMaxs)
        self.Done=False

    def GetPermutation(self):
        out={}

        i=0
        for p in self.keys:
            sys.stdout.flush()
            out[p]=self.Params[p][self.Digits.Digits[i]]
            i+=1

        if self.Digits.Increment():
            return out
        else:
            return False

    def Permutations(self):
        self.Digits.Reset()
        res=True
        
        out=[]
        while res:
            res=self.GetPermutation()
            out+=[res]

        self.Digits.Reset()
        return out[:-1]

