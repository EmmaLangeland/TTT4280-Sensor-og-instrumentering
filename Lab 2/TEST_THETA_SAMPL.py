import numpy as np

#Stoler ikke på denne heller:)
def theta(n_21, n_31, n_32):
    if ((n_31 - n_21 + 2*n_32) >= 0):
        theta = (np.arctan2(np.sqrt(3) * (n_31 + n_21), ((n_31 - n_21 + 2*n_32) + 1e-10))) # legger til + 1e-10 for å ikke dele på 0 i arctan funk
    else:
        theta = (np.arctan2(np.sqrt(3) * (n_31 + n_21), ((n_31 - n_21 + 2*n_32) + 1e-10))) + np.pi

    theta = theta * (180/np.pi)  
    #print(f"Innfallsvinkelen theta er på {theta} grader\n\n\n")
    return theta

v_0 = theta(-1,1,5) 
v_360 = theta(1,-1,-5) 
v_m90 = theta(-3,-3,0)
v_90 = theta(3,3,0)
v_m30 = theta(-4,-1,5) #Skal være 30, ble 240 m/4,-1,5, feil fortegn, byttet til -4,-1,-5, fikk -30
v_33 = theta(0,5,4) # Gir samme verdi for + og - 5 og 4
v_330 = theta(5,0,-5) # skal være ca. 210, men ga 330
v = theta(0,-5,-5) # skal være ca. 150, men ga 30

v = theta(0,5,5) # skal være ca. 210, men ga 330

print(v)

#Test for all posible combinations
#for i in range(-6,6,1):
    #for j in range(-6,6,1):
        #for k in range(-6,6,1):
            #v = theta(i,j,k)
            #print("Vinkel",v,"n21,n31,n32: (",i, ", ",j, ", ",k,")")