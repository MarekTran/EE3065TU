import numpy as np

def COPT_function(Cgens, Ugens, P_round):
    # Eduardo Jerez, based on Bart Tuinema's Matlab code
    # COPT calculation
    # COPT = COPT_function(Cgens,Ugens,P_round)
    # Cgens = unit capacities
    # Ugens = Generator unavailabilities (FOR's)
    # P_round = capacities are rounded to this value

    import numpy as np

    Cgens = np.multiply(P_round, np.round(np.divide(Cgens, P_round)))  # round capacities
    P_total = sum(Cgens)  # total capacity
    n_units = len(Cgens)  # number of units

    COPT = np.zeros([np.int_(P_total / P_round + 1), 2])
    COPT[:, 0] = range(0, np.int_(P_total) + P_round, P_round)
    COPT[0, 1] = 1

    for i in range(n_units):
        COPT2 = np.multiply(COPT[:, 1], Ugens[i])  # COPT when unit i is off
        COPT[:, 1] = np.multiply(COPT[:, 1], (1 - Ugens[i]))  # COPT when unit i is on
        COPT[:, 1] = np.add(COPT[:, 1], np.hstack([np.zeros([np.int_(Cgens[i] / P_round)]), COPT2[0:-np.int_(
            (Cgens[i] / P_round))]]))  # COPT2 is shifted down by the capacity of the generator

    COPT = np.transpose(np.vstack([np.transpose(COPT), np.subtract(1, np.hstack([[0], np.cumsum(COPT[0:-1, 1])]))]))

    return COPT
#Cgens =[200, 100, 200, 500]
#Ugens = [0.05, 0.03, 0.04, 0.06]

df = pd.read_csv('Cgens_Ugens.csv')
Cgens = df['Cgens'].values.tolist()
Ugens = df['Ugens'].values.tolist()

#print(COPT_function(Cgens, Ugens,2))

def LOLP(C, L, time):
    Ctotal = np.sum(C)
    Gen = []
    p = []
    P = []
    table = COPT_function(Cgens, Ugens, 100)
    for row in table:
        Gen.append(row[0])
        p.append(row[1])
        P.append(row[2])
    
    lolp = []
    for j in range(0,len(L)):
        for i in range(0,len(Gen)):
            if(Ctotal-Gen[i]<L[j]):
                lolp.append(time[j]*P[i])
                break
            else:
                continue
            
    return(np.sum(lolp)/np.sum(time))

def LOLE(LOLP, time):
    return np.sum(time)*LOLP

def LOEE(C, L, time):
    Ctotal = np.sum(C)
    Gen = []
    p = []
    P = []
    table = COPT_function(Cgens, Ugens, 100)
    for row in table:
        Gen.append(row[0])
        p.append(row[1])
        P.append(row[2])
        
    loee = []
    
    for j in range(0,len(L)):
        stamp =[]
        for i in range(0,len(Gen)):
            if(Ctotal-Gen[i]<L[j]):
                stamp.append((L[j]-(Ctotal-Gen[i]))*p[i])
            else:
                continue
        loee.append(np.sum(stamp)*time[j])
            
    return np.sum(loee)
    


lolp =  LOLP(Cgens, [550,350],[5000,3760])
print("THe LOLP of this COPT is: ", lolp)
print("THe LOLE of this COPT is: ", LOLE(lolp, [5000,3760]))
print("THe LOEE of this COPT is: ", LOEE(Cgens, [550,350], [5000,3760]), "MW")

        
