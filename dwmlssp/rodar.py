# with is like your try .. finally block in this case
import os
import re
with open('Config_1902.run', 'r') as f:
    # read a list of lines into data
    data = f.readlines()
    rodar = []
    for root, dirs, files in os.walk('Instancias'):
        for f in files:
             inst, ext =  f.split('.')
	     par = re.findall("\d+",inst)	
	     if (ext =='dat'):
		 par = re.findall("\d+",inst)
		 par = map(int,par)
		 if (par[0]*par[2]> 700):
		     tfo = 200
		     tlb = 50
		     klb = 15
                 elif (par[0]*par[2]> 400):
		     tfo = 150
		     tlb = 40
		     klb = 10
                 elif (par[0]*par[2]> 200):
		     tfo = 100
		     tlb = 30
		     klb = 8
                 else:
		     tfo = 50
		     tlb = 20
		     klb = 5			 	
                 data[9] = 'let inst:= "' + inst + '";\n'
                 #data[12] = 'let tlimFO:=' + str(tfo) + ';\n'
                 #data[13] = 'let tlimLB:=' + str(tlb) + ';\n'
                 #data[14] = 'let KLB:=' + str(klb) + ';\n'
                 run_inst= 'Config_'+inst+'.run'
		 rodar.append('ampl '+run_inst+'\n')
                 # and write everything back
                 with open(run_inst, 'w') as f:
                     f.writelines( data )
    with open('rodar_win.bat', 'w') as f:
        f.writelines(rodar)
