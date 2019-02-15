# -*- coding: utf-8 -*-
#Elaborado por Robson Vieira de Oliveira, 2018.

import random
from gurobipy import *
import math
import pickle
from time import time
from random import randrange, uniform

class Instancia:
    def __init__(self, P,F,Pf,Fp,s0,n,D,LM,FM,LT,lambd,sigma,delta,beta,alpha,delta_fix,fname,fname2):	
        self.P = P
        self.F = F
        self.Pf = Pf
        self.Fp = Fp
        self.s0 = s0
        self.n = n
        self.D = D
        self.LM = LM
        self.FM = FM
        self.LT = LT
        self.lambd = lambd
        self.sigma = sigma
        self.delta = delta
        self.beta = beta
        self.alpha = alpha
        self.delta_fix = delta_fix

def loadPickledInst(NomeSaida):
    with open(NomeSaida, 'r') as f:
	I = pickle.load(f)
    return I 

def calc_gamma(s0,n,D,LM,LT,beta,Fp,P,F):
    K = range(0,n)
    gamma = [[['.' for t in K]for f in F]for p in P]    
    for p in P:
        pi = P.index(p)
        a2 = sum(D[pi][t] for t in K) - s0[pi]
        for f in Fp[pi]:
            fi = F.index(f)
            for t in K:  
                a1 = sum(D[pi][t + LT[pi][fi]:])
                a3 = (float(min(a1,a2))/float(beta[pi][fi]))
                a4 = math.ceil(a3)
                gamma[pi][fi][t] = max(LM[pi][fi], a4)
    return gamma
    
def variaveis(a,n,F,Pf,Fp,P,colunas,o): #leitor dos valores das variáveis que saem do mestre
    K = range(n)
    w = 0 
    Z,L,FI = [],[],[]    
    lambd = []
    A1,A2,A3,A4,A5,A6,A7 = [],[],[],[],[],[],[]
    for f in F:
        zl = []      
        for t in K:
            zl.append(a[w])
            w+=1
        Z.append(zl)   
    for f in F:
        ll = []
        for t in K:
            ll.append(a[w])
            w+=1
        L.append(ll)
    for f in F:
        fil = []        
        for t in K:
            fil.append(a[w])
            w+=1
        FI.append(fil)    
    
    for p in P:
        pi = P.index(p)
        lambd1 = []
        for c in range(len(colunas[pi])):
            lambd1.append(a[w])
            w+=1
        lambd.append(lambd1)
    if o == 1:
        for p in P:
            pi = P.index(p)
            a1,a2 = [],[]   
            for f in Fp[pi]:
                a2=[]
                for t in K:
                    a2.append(a[w])
                    w+=1
                a1.append(a2)    
            A1.append(a1)
        for f in F:
            a2=[]
            for t in K:
                a2.append(a[w])
                w+=1
            A2.append(a2)    
        for p in P:
            pi = P.index(p)
            a1,a2 = [],[]   
            for f in Fp[pi]:
                a2=[]
                for t in K:
                    a2.append(a[w])
                    w+=1
                a1.append(a2)    
            A3.append(a1)         
        for f in F:
            a4=[]
            for t in K:
                a4.append(a[w])
                w+=1
            A4.append(a4)
        for f in F:
            a5=[]
            for t in K:
                a5.append(a[w])
                w+=1
            A5.append(a5)
        for p in P:
            A6.append(a[w])
            w+=1
        A7 = []    
        for p in P:
            A7.append(a[w])
            w+=1
        return Z,L,FI,lambd,A1,A2,A3,A4,A5,A6,A7
    return Z,L,FI,lambd
    
def variaveis2(a,n,P,F,Pf,Fp,p):
    K = range(n)
    w = 0
    pi = P.index(p)
    x,y = [],[]
    for f in Fp[pi]:
        ll = []
        for t in K:
            ll.append(a[w])
            w+=1
        x.append(ll)
    for f in Fp[pi]:
        yl2=[]
        for t in K:
            yl2.append(a[w])
            w+=1
        y.append(yl2)
    return x,y   

def duais(a,n,F,Pf,Fp,P):    
    w = 0 
    K = range(n)
    mi=[]
    for p in P:
        mi1,mi2=[],[]
        pi = P.index(p)      
        for f in Fp[pi]:
            mi2=[]
            for t in K: 
                mi2.append(a[w])
                w+=1
            mi1.append(mi2)
        mi.append(mi1)
    eta=[]    
    for f in F:
        eta1=[]
        for t in K: 
            eta1.append(a[w])
            w+=1
        eta.append(eta1)
    k=[]
    for p in P:
        k1,k2=[],[]
        pi = P.index(p)      
        for f in Fp[pi]:
            k2=[]
            for t in K: 
                k2.append(a[w])
                w+=1
            k1.append(k2)
        k.append(k1)
    u=[]    
    for f in F:
        u1=[]
        for t in K: 
            u1.append(a[w])
            w+=1
        u.append(u1)               
    r=[]    
    for f in F:
        r1=[]
        for t in K: 
            r1.append(a[w])
            w+=1
        r.append(r1)            
    PI = []        
    for p in P:
        PI.append(a[w])
        w+=1
    return mi,k,u,r,PI,eta
    
def problemamestre(P,F,Fp,Pf,s0,n,D,LT,theta,sigma,delta,beta,alpha,delta_fix,colunas,cont):
    time=0
    model = Model("Problema Mestre")
    #mp.setParam("MIPGap",0.0)
    model.setParam('OutputFlag', 0)
    #model.setParam('Presolve', 0)
    obj = 0
    K = range(n)
    X = [[[] for c in range(len(colunas[P.index(p)]))]for p in P] 
    Y = [[[] for c in range(len(colunas[P.index(p)]))]for p in P] 
    for p in P:
        pi = P.index(p)
        for c in range(len(colunas[pi])):
                X[pi][c] = colunas[pi][c][0]
                Y[pi][c] = colunas[pi][c][1]
    Z,L,FI = [],[],[]  
    lambd = []
    for f in F:
        fi = F.index(f)
        zl = []      
        for t in K:
            zl.append(model.addVar(vtype="C",lb=0.0,ub=1.0, name= 'Z'))
        Z.append(zl)   
    for f in F:
        fi = F.index(f)
        ll = []
        for t in K:
            ll.append(model.addVar(vtype="C", name= 'L'))
        L.append(ll)
    for f in F:
        fi = F.index(f)
        fil = []        
        for t in K:
            fil.append(model.addVar(vtype="C",lb=0.0,ub=1.0, name= 'FI'))
        FI.append(fil)    
    for p in P:
        pi = P.index(p)
        lambd1 = []
        for c in range(len(colunas[pi])):
            lambd1.append(model.addVar(vtype="C", name = 'Lambda'))
        lambd.append(lambd1)
    model.update()
    for p in P:
        pi = P.index(p)                                                                  
        for f in Fp[pi]:
            fi = F.index(f)
            fa = Fp[pi].index(f)
            for t in K:
                model.addConstr(quicksum(X[pi][c][fa][t]*lambd[pi][c] for c in range(len(colunas[pi])) ) <= quicksum(gamma[pi][fi][t] * Y[pi][c][fa][t] *lambd[P.index(p)][c] for c in range(len(colunas[pi])) ))
    for f in F:
        fi = F.index(f)
        for t in K: 
            model.addConstr(Z[fi][t] >= (1/float(len(Pf[fi])))*quicksum(Y[P.index(p)][c][Fp[P.index(p)].index(f)][t] *lambd[P.index(p)][c] for p in Pf[fi] for c in range(len(colunas[P.index(p)])))) #relação zy           
    for p in P:
        pi = P.index(p)                                                                  
        for f in Fp[pi]:
            fi = F.index(f)
            fa = Fp[pi].index(f)
            for t in K:
                model.addConstr(quicksum( X[pi][c][fa][t]*lambd[pi][c] for c in range(len(colunas[pi])) ) >= quicksum(LM[pi][fi] * Y[pi][c][fa][t] *lambd[P.index(p)][c] for c in range(len(colunas[pi])) ) ) #lote minimo
    for f in F:
        fi = F.index(f)
        for t in K:               
            model.addConstr(quicksum( (sigma[P.index(p)][fi]*X[P.index(p)][c][Fp[P.index(p)].index(f)][t])*lambd[P.index(p)][c] for p in Pf[fi] for c in range(len(colunas[P.index(p)])) if t+LT[P.index(p)][fi] < n) >= FM[fi]*(Z[fi][t] - FI[fi][t])) #fatura minima
    for f in F: 
        fi = F.index(f)
        for t in K:            
            model.addConstr(L[fi][t] - quicksum((X[P.index(p)][c][Fp[P.index(p)].index(f)][t] * alpha[P.index(p)][fi])*lambd[P.index(p)][c] for p in Pf[fi] for c in range(len(colunas[P.index(p)])) if t+LT[P.index(p)][fi] < n)  - (quicksum(FI[fi][t]*gamma[P.index(p)][fi][t]*alpha[P.index(p)][fi]for p in Pf[fi] if t+LT[P.index(p)][F.index(f)] < n)) >= - (quicksum(gamma[P.index(p)][fi][t]*alpha[P.index(p)][fi]for p in Pf[fi] if t+LT[P.index(p)][F.index(f)] < n))) #frete
    for p in P:
        pi = P.index(p)
        model.addConstr(quicksum(lambd[pi][c] for c in range(len(colunas[pi]))) == 1)
    model.setObjective(quicksum((sigma[P.index(p)][F.index(f)]*X[P.index(p)][c][Fp[P.index(p)].index(f)][t])*lambd[P.index(p)][c] for p in P for f in Fp[P.index(p)] for t in K for c in range(len(colunas[P.index(p)])) if t+LT[P.index(p)][F.index(f)] < n) + quicksum((L[F.index(f)][t] * delta[F.index(f)] for f in F for t in K)) + quicksum(delta_fix[F.index(f)] * FI[F.index(f)][t] for f in F for t in K),GRB.MINIMIZE)       
    model.update()
    model.optimize()
    time=model.Runtime
    obj = model.objVal
    a = []
    for c in model.getConstrs():
        a.append(c.getAttr("Pi")) 
    mi,k,u,r,PI,eta = duais(a,n,F,Pf,Fp,P) 
    var=[]
    for c in model.getVars():
        var.append(c.x) 
    return mi,k,u,r,PI,obj,time,var,eta
    
def subproblema(P,F,Fp,Pf,n,D,LM,FM,LT,sigma,delta,beta,alpha,delta_fix,gamma,mi,k,u,r,eta,PI,cont):
    K = range(0,n)
    M = range(1,n)
    #g = k[:]
    time=0
    obj = 0
    objv = []
    Coluna = []
    col_entrou = []
    nova_coluna = 0
    for p in P:
        pi = P.index(p)
        model = Model("SubProblema")
        model.setParam('Presolve', 0)
        model.setParam('MIPFocus', 3)
        #model.setParam('SolutionLimit', 3)
        model.setParam('OutputFlag', 0)
        coluninha = []    
        X,S,A = [],[],[]
        for f in Fp[pi]:
            xl1=[]
            for t in K:
               xl1.append(model.addVar(vtype="I", name='X['+str(p)+','+str(f)+','+str(t)+']'))
            X.append(xl1) 
        for f in Fp[pi]:
            j1=[]
            for t in K:
               j1.append(model.addVar(vtype="B", name='A['+str(p)+','+str(f)+','+str(t)+']'))
            A.append(j1)
        for t in K:
            S.append(model.addVar(vtype="C", name= 'S['+str(p)+','+str(t)+']'))
        model.update()
        model.addConstr(S[0] == s0[pi] - D[pi][0] + quicksum(X[Fp[pi].index(f)][0]*beta[pi][F.index(f)]  for f in Fp[pi] if LT[pi][F.index(f)] == 0)) #demanda 0
        for t in M:
            model.addConstr(S[t] == S[t-1] - D[pi][t] + quicksum(X[Fp[pi].index(f)][t-LT[pi][F.index(f)]]*beta[pi][F.index(f)] for f in Fp[pi] if t - LT[pi][F.index(f)] >= 0)) #demanda
        for f in Fp[pi]:
            fi = F.index(f)
            fa = Fp[pi].index(f)
            for t in K:
                model.addConstr(X[fa][t] >= (LM[pi][fi] * A[fa][t])) 
        for f in Fp[pi]:
            fi = F.index(f)
            fa = Fp[pi].index(f)
            for t in K:
                model.addConstr(X[fa][t] <= (gamma[pi][fi][t] * A[fa][t]))
        model.setObjective(quicksum((theta[pi] * S[t]) for t in K ) + quicksum((mi[pi][Fp[pi].index(f)][t]*gamma[pi][F.index(f)][t] + eta[F.index(f)][t] + LM[pi][F.index(f)]*k[pi][Fp[pi].index(f)][t])*A[Fp[pi].index(f)][t] for f in Fp[pi] for t in K) + quicksum((sigma[pi][F.index(f)] - mi[pi][Fp[pi].index(f)][t] - k[pi][Fp[pi].index(f)][t] - sigma[pi][F.index(f)]*u[F.index(f)][t] + alpha[pi][F.index(f)]*r[F.index(f)][t])*X[Fp[pi].index(f)][t] for f in Fp[pi] for t in K if t+LT[P.index(p)][F.index(f)] < n) - PI[pi] ,GRB.MINIMIZE)       
        model.update() 
        model.optimize()
        time+=model.Runtime
        obj = model.objVal
        if round(obj,4) < 0:  
                objv.append(obj)
                nova_coluna += 1
                a = []
                col_entrou.append((p))
                for v in model.getVars():
                    a.append(v.x)
                x,y = variaveis2(a,n,P,F,Pf,Fp,p)
                coluninha.append(x)
                coluninha.append(y)
                Coluna.append((pi,coluninha)) 
    return Coluna,objv,nova_coluna,col_entrou,time,obj
        
def problemamestre_artificial(P,F,Fp,Pf,s0,n,D,LT,theta,sigma,delta,beta,alpha,delta_fix,colunas,cont):
    time=0
    model = Model("Problema Mestre")
    #mp.setParam("MIPGap",0.0)
    model.setParam('OutputFlag', 0)
    obj = 0
    K = range(n)
    X = [[[] for c in range(len(colunas[P.index(p)]))]for p in P] 
    Y = [[[] for c in range(len(colunas[P.index(p)]))]for p in P] 
    for p in P:
        pi = P.index(p)
        for c in range(len(colunas[pi])):
                X[pi][c] = colunas[pi][c][0]
                Y[pi][c] = colunas[pi][c][1]
    Z,L,FI = [],[],[]    
    lambd = []
    A1,A2,A3,A4,A5,A6,A7 = [],[],[],[],[],[],[]
    for f in F:
        fi = F.index(f)
        zl = []      
        for t in K:
            zl.append(model.addVar(vtype="C",lb=0.0,ub=1.0, name= 'Z['+str(f)+','+str(t)+']'))
        Z.append(zl)   
    for f in F:
        fi = F.index(f)
        ll = []
        for t in K:
            ll.append(model.addVar(vtype="C", name= 'L['+str(f)+','+str(t+1)+']'))
        L.append(ll)
    for f in F:
        fi = F.index(f)
        fil = []        
        for t in K:
            fil.append(model.addVar(vtype="C",lb=0.0,ub=1.0, name= 'FI['+str(f)+','+str(t+1)+']'))
        FI.append(fil)    
    for p in P:
        pi = P.index(p)
        lambd1 = []
        for c in range(len(colunas[pi])):
            lambd1.append(model.addVar(vtype="C", name = 'Lambda['+str(p)+',Col '+str(c)+']'))
        lambd.append(lambd1)
    for p in P:
        pi = P.index(p)
        a1,a2 = [],[]   
        for f in Fp[pi]:
            a2=[]
            for t in K:
                a2.append(model.addVar(vtype="C"))
            a1.append(a2)    
        A1.append(a1)
    for f in F:
        a2=[]
        for t in K:
            a2.append(model.addVar(vtype="C"))
        A2.append(a2)    
    for p in P:
        pi = P.index(p)
        a1,a2 = [],[]   
        for f in Fp[pi]:
            a2=[]
            for t in K:
                a2.append(model.addVar(vtype="C"))
            a1.append(a2)    
        A3.append(a1)         
    for f in F:
        a4=[]
        for t in K:
            a4.append(model.addVar(vtype="C"))
        A4.append(a4)
    for f in F:
        a5=[]
        for t in K:
            a5.append(model.addVar(vtype="C"))
        A5.append(a5)
    for p in P:
        A6.append(model.addVar(vtype="C"))
    for p in P:
        A7.append(model.addVar(vtype="C"))
    model.update()
    for p in P:
        pi = P.index(p)                                                                  
        for f in Fp[pi]:
            fi = F.index(f)
            fa = Fp[pi].index(f)
            for t in K:
                model.addConstr(quicksum(X[pi][c][fa][t]*lambd[pi][c] for c in range(len(colunas[pi])) ) + A1[pi][fa][t] <= quicksum(gamma[pi][fi][t] * Y[pi][c][fa][t] *lambd[P.index(p)][c] for c in range(len(colunas[pi])) )) #relaçãoXY 
    for f in F:
        fi = F.index(f)
        for t in K: 
            model.addConstr(Z[fi][t] + A2[fi][t]>= (1/float(len(Pf[fi])))*quicksum(Y[P.index(p)][c][Fp[P.index(p)].index(f)][t] *lambd[P.index(p)][c] for p in Pf[fi] for c in range(len(colunas[P.index(p)])))) #relação zy                   
    for p in P:
        pi = P.index(p)                                                                  
        for f in Fp[pi]:
            fi = F.index(f)
            fa = Fp[pi].index(f)
            for t in K:
                model.addConstr(quicksum( X[pi][c][fa][t]*lambd[pi][c] for c in range(len(colunas[pi])) ) + A3[pi][fa][t] >=  quicksum(LM[pi][fi] * Y[pi][c][fa][t] *lambd[P.index(p)][c] for c in range(len(colunas[pi])) ) ) #lote minimo
    for f in F:
        fi = F.index(f)
        for t in K:               
            model.addConstr(quicksum( (sigma[P.index(p)][fi]*X[P.index(p)][c][Fp[P.index(p)].index(f)][t])*lambd[P.index(p)][c] for p in Pf[fi] for c in range(len(colunas[P.index(p)])) if t+LT[P.index(p)][F.index(f)] < n) + A4[fi][t]  >= FM[fi]*(Z[fi][t] - FI[fi][t])) #fatura minima
    for f in F: 
        fi = F.index(f)
        for t in K:            
            model.addConstr(L[fi][t] - quicksum(FI[fi][t]*gamma[P.index(p)][fi][t]*alpha[P.index(p)][fi]for p in Pf[fi] if t+LT[P.index(p)][F.index(f)] < n) + A5[fi][t] >= quicksum((X[P.index(p)][c][Fp[P.index(p)].index(f)][t] * alpha[P.index(p)][fi])*lambd[P.index(p)][c] for p in Pf[fi] for c in range(len(colunas[P.index(p)])) if t+LT[P.index(p)][F.index(f)] < n) - (quicksum(gamma[P.index(p)][fi][t]*alpha[P.index(p)][fi]for p in Pf[fi] if t+LT[P.index(p)][F.index(f)] < n))) #frete
    for p in P:
        pi = P.index(p)
        model.addConstr(quicksum(lambd[pi][c] for c in range(len(colunas[pi]))) + A6[pi] - A7[pi] == 1)
    model.setObjective(quicksum(A1[P.index(p)][Fp[P.index(p)].index(f)][t]for p in P for f in Fp[P.index(p)] for t in K) + quicksum(A2[F.index(f)][t] for f in F for t in K)+quicksum(A3[P.index(p)][Fp[P.index(p)].index(f)][t]for p in P for f in Fp[P.index(p)] for t in K) + quicksum(A4[F.index(f)][t] for f in F for t in K)+ quicksum(A5[F.index(f)][t] for f in F for t in K)+ quicksum(A6[P.index(p)]  for p in P) + quicksum(A7[P.index(p)]  for p in P),GRB.MINIMIZE)       
    model.update()
    model.optimize()
    time=model.Runtime
    obj = model.objVal
    print obj
    a = []
    for c in model.getConstrs():
        a.append(c.getAttr("Pi")) 
    mi,k,u,r,PI,eta = duais(a,n,F,Pf,Fp,P) #criar
    var=[]
    for c in model.getVars():
        var.append(c.x) 
    return mi,k,u,r,PI,obj,time,var,eta
    
def subproblema_artificial(P,F,Fp,Pf,n,D,LM,FM,LT,sigma,delta,beta,alpha,delta_fix,gamma,mi,k,u,r,eta,PI,cont):
    K = range(0,n)
    M = range(1,n)
    time=0
    obj = 0
    objv = []
    Coluna = []
    col_entrou = []
    nova_coluna = 0
    for p in P:
        pi = P.index(p)
        model = Model("SubProblema")
        #model.setParam('Presolve', 0)
        #model.setParam('SolutionLimit', 1)
        model.setParam('MIPFocus', 3)
        model.setParam('OutputFlag', 0)
        coluninha = [] 
        X,S,A = [],[],[]
        for f in Fp[pi]:
            xl1=[]
            for t in K:
               xl1.append(model.addVar(vtype="I", name='X['+str(p)+','+str(f)+','+str(t)+']'))
            X.append(xl1) 
        for f in Fp[pi]:
            j1=[]
            for t in K:
               j1.append(model.addVar(vtype="B", name='A['+str(p)+','+str(f)+','+str(t)+']'))
            A.append(j1)
        for t in K:
            S.append(model.addVar(vtype="C", name= 'S['+str(p)+','+str(t)+']'))
        model.update()
        model.addConstr(S[0] == s0[pi] - D[pi][0] + quicksum(X[Fp[pi].index(f)][0]*beta[pi][F.index(f)]  for f in Fp[pi] if LT[pi][F.index(f)] == 0)) #demanda 0
        for t in M:
            model.addConstr(S[t] == S[t-1] - D[pi][t] + quicksum(X[Fp[pi].index(f)][t-LT[pi][F.index(f)]]*beta[pi][F.index(f)] for f in Fp[pi] if t - LT[pi][F.index(f)] >= 0)) #demanda
        for f in Fp[pi]:
            fi = F.index(f)
            fa = Fp[pi].index(f)
            for t in K:
                model.addConstr(X[fa][t] >= (LM[pi][fi] * A[fa][t])) #relaçãoXY 
        for f in Fp[pi]:
            fi = F.index(f)
            fa = Fp[pi].index(f)
            for t in K:
                model.addConstr(X[fa][t] <= (gamma[pi][fi][t] * A[fa][t]))
        model.setObjective(quicksum((theta[pi] * S[t]) for t in K ) + quicksum((mi[pi][Fp[pi].index(f)][t]*gamma[pi][F.index(f)][t] + eta[F.index(f)][t]*(1/(len(Pf[F.index(f)]))) + LM[pi][F.index(f)]*k[pi][Fp[pi].index(f)][t])*A[Fp[pi].index(f)][t] for f in Fp[pi] for t in K) + quicksum(( - mi[pi][Fp[pi].index(f)][t] - k[pi][Fp[pi].index(f)][t] - sigma[pi][F.index(f)]*u[F.index(f)][t] + alpha[pi][F.index(f)]*r[F.index(f)][t])*X[Fp[pi].index(f)][t] for f in Fp[pi] for t in K if t+LT[P.index(p)][F.index(f)] < n) - PI[pi] ,GRB.MINIMIZE)       
        model.update() 
        model.optimize()
        time+=model.Runtime
        obj = model.objVal
        objv.append(obj)
        if round(obj,4) < 0:            
            nova_coluna += 1
            a = []
            col_entrou.append((p))
            for v in model.getVars():
                a.append(v.x)
            x,y = variaveis2(a,n,P,F,Pf,Fp,p)
            coluninha.append(x)
            coluninha.append(y)
            Coluna.append((pi,coluninha)) 
    return Coluna,objv,nova_coluna,col_entrou,time
    
def MP(P,F,Fp,Pf,s0,n,D,LT,theta,sigma,delta,beta,alpha,delta_fix,colunas,cont,zmp,lmp,fimp,lambdmp):
    time=0
    model = Model("Problema Mestre")
    model.setParam('OutputFlag', 0)
    obj = 0
    K = range(n)
    M = range(1,n)
    X = [[[] for c in range(len(colunas[P.index(p)]))]for p in P] 
    Y = [[[] for c in range(len(colunas[P.index(p)]))]for p in P] 
    for p in P:
        pi = P.index(p)
        for c in range(len(colunas[pi])):
                X[pi][c] = colunas[pi][c][0]
                Y[pi][c] = colunas[pi][c][1]
    Z=zmp[:]
    L=lmp[:]
    FI = fimp[:]    
    lambd = lambdmp[:]
    S=[]   
    for p in P:
        pi = P.index(p)
        sl = []
        for t in K:
            sl.append(model.addVar(vtype="C", name= 'S['+str(p)+','+str(t+1)+']'))
        S.append(sl)
    model.update()
    for p in P:
        pi = P.index(p)
        model.addConstr(S[pi][0] == s0[pi] - D[pi][0] + quicksum((X[pi][c][Fp[pi].index(f)][0]*beta[pi][F.index(f)])*lambd[pi][c] for f in Fp[pi] for c in range(len(colunas[pi])) if LT[pi][F.index(f)] == 0)) #demanda 0
    for p in P:
        pi = P.index(p)           
        for t in M:
            model.addConstr(S[pi][t] == S[pi][t-1] - D[pi][t] + quicksum((X[pi][c][Fp[pi].index(f)][t-LT[pi][F.index(f)]]*beta[pi][F.index(f)])*lambd[pi][c] for f in Fp[pi] for c in range(len(colunas[pi]))  if t - LT[pi][F.index(f)] >= 0)) #demanda
    for p in P:
        pi = P.index(p)                                                                  
        for f in Fp[pi]:
            fi = F.index(f)
            fa = Fp[pi].index(f)
            for t in K:
                model.addConstr(quicksum(X[pi][c][fa][t]*lambd[pi][c] for c in range(len(colunas[pi])) ) <= quicksum(gamma[pi][fi][t] * Y[pi][c][fa][t] *lambd[P.index(p)][c] for c in range(len(colunas[pi])) ))
    for f in F:
        fi = F.index(f)
        for t in K: 
            model.addConstr(Z[fi][t] >= (1/float(len(Pf[fi])))*quicksum(Y[P.index(p)][c][Fp[P.index(p)].index(f)][t] *lambd[P.index(p)][c] for p in Pf[fi] for c in range(len(colunas[P.index(p)])))) #relação zy           
    for p in P: 
        pi = P.index(p)                                                                  
        for f in Fp[pi]:
            fi = F.index(f)
            fa = Fp[pi].index(f)
            for t in K:
                model.addConstr(quicksum( X[pi][c][fa][t]*lambd[pi][c] for c in range(len(colunas[pi])) ) >= quicksum(LM[pi][fi] * Y[pi][c][fa][t] *lambd[P.index(p)][c] for c in range(len(colunas[pi])) ) ) #lote minimo
    for f in F:
        fi = F.index(f)
        for t in K:               
            model.addConstr(quicksum( (sigma[P.index(p)][fi]*X[P.index(p)][c][Fp[P.index(p)].index(f)][t])*lambd[P.index(p)][c] for p in Pf[fi] for c in range(len(colunas[P.index(p)])) ) >= FM[fi]*(Z[fi][t] - FI[fi][t])) #fatura minima
    for f in F:
        fi = F.index(f)
        for t in K:            
            model.addConstr(L[fi][t] - quicksum((X[P.index(p)][c][Fp[P.index(p)].index(f)][t] * alpha[P.index(p)][fi])*lambd[P.index(p)][c] for p in Pf[fi] for c in range(len(colunas[P.index(p)])) )  - (quicksum(FI[fi][t]*gamma[P.index(p)][fi][t]*alpha[P.index(p)][fi]for p in Pf[fi])) >= - (quicksum(gamma[P.index(p)][fi][t]*alpha[P.index(p)][fi]for p in Pf[fi]))) #frete
    for p in P:
        pi = P.index(p)
        model.addConstr(quicksum(lambd[pi][c] for c in range(len(colunas[pi]))) == 1)
    model.setObjective(quicksum((theta[P.index(p)] * S[P.index(p)][t])*lambd[P.index(p)][c] for p in P for t in K for c in range(len(colunas[P.index(p)])) ) + quicksum((sigma[P.index(p)][F.index(f)]*X[P.index(p)][c][Fp[P.index(p)].index(f)][t])*lambd[P.index(p)][c] for p in P for f in Fp[P.index(p)] for t in K for c in range(len(colunas[P.index(p)])) if t+LT[P.index(p)][F.index(f)] < n) + quicksum((L[F.index(f)][t] * delta[F.index(f)] for f in F for t in K)) + quicksum(delta_fix[F.index(f)] * FI[F.index(f)][t] for f in F for t in K),GRB.MINIMIZE)       
    model.update()
    model.optimize()
    obj = model.objVal
    return obj  
                   
if __name__=="__main__":
    Pr=[40] 
    Fo=[25]
    T=[12]
    V=[0,1,2,3,4]
    for pr in Pr:
        for fo in Fo:
            for te in T:
                for v in V:
                    I = loadPickledInst('/home/servidor-lasos/Thiago/Robson/DW/dwmlssp/inst/py_inst_p'+str(pr)+'f'+str(fo)+'t'+str(te)+'_'+str(v)+'_bin.dat')
                    P = I.P
                    F = I.F
                    Pf = I.Pf
                    Fp = I.Fp
                    s0 = I.s0
                    n = I.n
                    D = I.D
                    LM = I.LM
                    FM = I.FM
                    LT = I.LT
                    theta = I.lambd
                    sigma = I.sigma
                    delta = I.delta
                    beta = I.beta
                    alpha = I.alpha
                    delta_fix = I.delta_fix
                    gamma = calc_gamma(s0,n,D,LM,LT,beta,Fp,P,F)#cálculo dos valores Big M
                    for p in P:#algumas instancias estao com o alpha como string, passei pra int com esse trecho
                        pi = P.index(p)
                        for f in F:
                            fi = F.index(f)
                            x = alpha[pi][fi]
                            a = 0
                            while x[a] != '.':
                                a +=1
                            if a != 0:
                                num = int(x[0:a])
                                z = int(x[a+1:])
                                z = z * 10e-3
                                num = num + z
                                alpha[pi][fi] = num
                    K = range(n)
                    obj2 = 1
                    coluns = [[] for p in P]
                    mi = [[[0 for t in K]for f in Fp[P.index(p)]] for p in P]
                    k = [[[0 for t in K]for f in Fp[P.index(p)]] for p in P]
                    r = [[0 for t in K]for f in F]
                    u = [[0 for t in K]for f in F]
                    eta = [[0 for t in K]for f in F]
                    PI = [1 for p in P] #inicia dual de convexidade com 1
                    cont1 = 1
                    cont2 = 1
                    tempo = 0
                    runtime = 0
                    strt = time()
                    var=[]
                    o = 1
                    while obj2 > 0.000000000000000001:
                        print cont1
                        col,obj,nova_col,col_entrou,runtime = subproblema_artificial(P,F,Fp,Pf,n,D,LM,FM,LT,sigma,delta,beta,alpha,delta_fix,gamma,mi,k,u,r,eta,PI,cont1) #gera uma coluna com o mi e pi
                        tempo+=runtime
                        if nova_col == 0:
                            break
                        z = 0
                        while z < len(col):
                            coluns[col[z][0]].append(col[z][1])
                            z+=1
                        mi,k,u,r,PI,obj2,runtime,var,eta = problemamestre_artificial(P,F,Fp,Pf,s0,n,D,LT,theta,sigma,delta,beta,alpha,delta_fix,coluns,cont1)#cria novos duais
                        zmp,lmp,fimp,lambdmp,a1,a2,a3,a4,a5,a6,a7 = variaveis(var,n,F,Pf,Fp,P,coluns,o)
                        tempo+=runtime
                        cont1 += 1 
                                           
                    while True:
                        o = 0
                        mi,k,u,r,PI,obj3,runtime,var,eta = problemamestre(P,F,Fp,Pf,s0,n,D,LT,theta,sigma,delta,beta,alpha,delta_fix,coluns,cont2)
                        zmp,lmp,fimp,lambdmp = variaveis(var,n,F,Pf,Fp,P,coluns,o)
                        tempo+=runtime
                        col,obj,nova_col,col_entrou,runtime,obj4 = subproblema(P,F,Fp,Pf,n,D,LM,FM,LT,sigma,delta,beta,alpha,delta_fix,gamma,mi,k,u,r,eta,PI,cont2)#utiliza os novos duais para gerar uma nova coluna
                        tempo+=runtime
                        z = 0
                        while z < len(col):
                            coluns[col[z][0]].append(col[z][1])
                            z+=1
                                                    
                        if nova_col == 0:
                            doc = open('/home/servidor-lasos/Thiago/Robson/DW/dwmlssp/result/resultados'+str(pr)+'f'+str(fo)+'t'+str(te)+'_'+str(v)+'.txt', 'w')
                            doc.write('RESOLUÇÃO INSTANCIA' +str(pr)+'f'+str(fo)+'t'+str(te)+'_'+str(v)+'\n\n')
                            doc.close()
                            doc = open('/home/servidor-lasos/Thiago/Robson/DW/dwmlssp/result/resultados'+str(pr)+'f'+str(fo)+'t'+str(te)+'_'+str(v)+'.txt', 'w')
                            zmp,lmp,fimp,lambdmp = variaveis(var,n,F,Pf,Fp,P,coluns,o)
                            obj5 = MP(P,F,Fp,Pf,s0,n,D,LT,theta,sigma,delta,beta,alpha,delta_fix,coluns,cont2,zmp,lmp,fimp,lambdmp)
                            doc.write("\nValores das variáveis z\n")
                            for x in zmp:
                                doc.write("%s\n" %x)
                            doc.write("\nValores das variáveis l\n")
                            for x in lmp:
                                doc.write("%s\n" %x)
                            doc.write("\nValores das variáveis fi\n")
                            for x in fimp:
                                doc.write("%s\n" %x)
                            doc.write("\nValores das variáveis lambda\n")
                            for x in lambdmp:
                                doc.write("%s\n" %x)
                            doc.write("\nValores dos duais (mi)\n")
                            for x in mi:
                                doc.write("%s\n" %x)
                            doc.write("\nValores dos duais (eta)\n")
                            for x in eta:
                                doc.write("%s\n" %x)
                            doc.write("\nValores dos duais (k)\n")
                            for x in k:
                                doc.write("%s\n" %x)
                            doc.write("\nValores dos duais (u)\n")
                            for x in u:
                                doc.write("%s\n" %x)
                            doc.write("\nValores dos duais (r)\n")
                            for x in r:
                                doc.write("%s\n" %x)
                            doc.write("\nValores dos duais (PI)\n")
                            for x in PI:
                                doc.write("%s\n" %x)
                            X = [[[] for c in range(len(coluns[P.index(p)]))]for p in P] 
                            Y = [[[] for c in range(len(coluns[P.index(p)]))]for p in P] 
                            for p in P:
                                pi = P.index(p)
                                for c in range(len(coluns[pi])):
                                        X[pi][c] = coluns[pi][c][0]
                                        Y[pi][c] = coluns[pi][c][1]
                            x = [[[0 for t in K]for f in Fp[P.index(p)]]for p in P]
                            y = [[[0 for t in K]for f in Fp[P.index(p)]]for p in P]
                            for p in P:
                                pi = P.index(p)
                                for f in Fp[P.index(p)]:
                                    fa = Fp[pi].index(f)
                                    for t in K:
                                        x[pi][fa][t] = (quicksum(X[pi][c][fa][t]*lambdmp[pi][c] for c in range(len(coluns[P.index(p)]))))
                                        y[pi][fa][t] = (quicksum(Y[pi][c][fa][t]*lambdmp[pi][c] for c in range(len(coluns[P.index(p)]))))
                            doc.write("\nValores originais (x)\n\n")
                            for p in P:
                                pi = P.index(p)
                                for f in Fp[P.index(p)]:
                                    fa = Fp[pi].index(f)
                                    for t in K:
                                        doc.write("\nx[p"+str(pi+1)+"][f"+str(fa+1)+"][t"+str(t+1)+"]:%s\n" %x[pi][fa][t])
                            doc.write("\nValores originais (y)\n\n")
                            for p in P:
                                pi = P.index(p)
                                for f in Fp[P.index(p)]:
                                    fa = Fp[pi].index(f)
                                    for t in K:
                                        doc.write("\ny[p"+str(pi+1)+"][f"+str(fa+1)+"][t"+str(t+1)+"]:%s\n" %y[pi][fa][t])
                            doc.write("\n\nTempo total gasto é %.5f\n" %tempo)
                            doc.write("\n\nTempo total gasto real é %.5f\n" %( time() - strt ))
                            doc.write("\n\nFunção obj final %.5f\n" %obj5)
                            doc.write("\n\nNumero de iterações final: %d\n" %(cont1+cont2))
                            doc.write("\n\nNumero de iterações ARTIFICIAL: %d\n" %cont1)
                            doc.write("\n\nNumero de iterações DW: %d\n" %cont2)
                            doc.close()
                            break
                        cont2 += 1
