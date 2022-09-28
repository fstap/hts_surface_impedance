# © 2022 Finn Stapelfeldt
# https://github.com/fstap
import numpy as np
import scipy.constants as sc
from scipy.integrate import quad
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.lib.scimath import sqrt as csqrt
import time

class SurfImp_Algorithm:

    np.seterr(all='raise')
    sns.set_context("poster") 
    sns.set_style("ticks",{"xtick.direction" : u"in","ytick.direction" : u"in"})
    sns.set_palette(sns.color_palette("coolwarm", 4))

    #nach 10.1134/1.1326973
    def debye_integrand(self,z): #used to caclulate the bloch grüneisen integral
        return z**5.0*np.exp(z)/np.power(np.exp(z)-1.0,2)

    def calc_tau(self,calc_params,temp): #approx. 10^-11, looks good - see https://arxiv.org/pdf/1610.09501.pdf
        tau_tc = calc_params['tau_tc']
        tc = calc_params['tc']
        t = temp/tc
        beta = calc_params['beta'] #beta=1 for Nuss, beta=0.005 for ref
        result_kappa_t,err_kappa_t = quad(self.debye_integrand,0,9/t)
        result_kappa,err_kappa = quad(self.debye_integrand,0,9)
        return np.power(np.power(tau_tc,-1)*(beta+result_kappa_t/result_kappa * t**5.0)/(beta+1.0),-1)

    def calc_ns_n(self,calc_params,temp): #more complex ns/n calculation. simple calculation methods yields nicer results!
        t = temp/calc_params['tc']
        alpha = calc_params['alpha']
        delta = calc_params['delta']
        if(t<1):
            return ( np.power(1.0-t,alpha) * (1.0-delta) + delta*(1.0 - np.power(t, 1.0/delta)**4))
        else:
            print('Sweep temperature value larger than Tc. This model cannot describe such values. Exiting.')
            exit()

    def calc_n(self,london_depth_0): #calculate n from london depth at 0K. at 0K, ns = n, nn is zero (at least approximately!) https://arxiv.org/pdf/1610.09501.pdf, tinkham-m-introduction-to-superconductivity p. 434
        return sc.m_e/(sc.e**2*sc.mu_0*london_depth_0**2)

    def calc_ns_n_simple(self,calc_params,temp): #prefered calculation method for ns/n
        t = temp/calc_params['tc']
        alpha = calc_params['alpha']
        #if(t<1):
        #    return np.power(1.0-t,0.5) #even more basic version, not too great..
        #else:
        #    return 1
        return 1-alpha*t-(1-alpha)*t**6

    def coth(self,x): #hyperbolic cotangent of x 
        return np.cosh(x)/np.sinh(x)

    def london_depth_temp_correct(self,calc_params,temp):
        alpha = calc_params['alpha']
        Tc = calc_params['tc']
        t = temp/Tc
        return calc_params['london_depth_0']/csqrt(1-alpha*t-(1-alpha)*t**6)
        #return 1/csqrt(1-t)*london_depth_0

    def calc_Z_jj(self,calc_params,temp_range):
        Zjj = []
        Tc = calc_params['tc']
        cond_thickness = calc_params['cond_thickness']
        cohrenece_length = 2.0e-9 #https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwihr--9_YryAhUpM-wKHZ-cATkQFjAAegQIBBAD&url=https%3A%2F%2Fmediatum.ub.tum.de%2Fdoc%2F602971%2F602971.pdf&usg=AOvVaw1FvDu-6DCw5dBwb-mDKNGl
        e_r_ybco = 5 #Epsillon_r von Ybco auf grain boundary
        for temp in temp_range:
            #london_depth = 1/(sc.mu_0*2*sc.pi*freq)*csqrt(np.imag(Zs)**2-np.real(Zs)**2)#https://sci-hub.st/10.1103/physrevb.49.9924
            
            london_depth = self.london_depth_temp_correct(calc_params,temp)
            #skin_depth = lambda_l_omega/csqrt(2.0j*2*sc.pi*freq*tau/(1+1j*2*sc.pi*freq*tau)) #https://arxiv.org/pdf/1610.09501.pdf
            #skin_depth = csqrt(2)*lambda_l_omega/csqrt(2.0*sc.pi*freq*tau*(nn_n/ns_n)-1j) #SurfImpAlgorithm
            #reinhold kleiner
            c_cbar = csqrt((cohrenece_length/(e_r_ybco*(cohrenece_length+london_depth*(1+self.coth(cond_thickness/london_depth)))))**(-1))
            Zjj.append(c_cbar*(cohrenece_length/(2e-6*e_r_ybco))*376.7303)
            #print(f'{temp},{c_cbar*(cohrenece_length/(2e-6*e_r_ybco))*376.7303}')
            #print(str(temp) + ' ' + str(london_depth))
        return Zjj

    def calc_sigma(self,freq,temp_range,calc_params):
        for param_name,param_value in calc_params.items():
            assert isinstance(param_value, float) or isinstance(param_value, np.float64), param_name + ' must be a float'    
        
        sigma1 = []
        sigma2 = []

        for temp in temp_range:
            tau = self.calc_tau(calc_params,temp)
            ns_n = self.calc_ns_n_simple(calc_params,temp)
            nn_n = 1.0-ns_n

            n = self.calc_n(calc_params['london_depth_0'])
            
            sigma1_t = (n*nn_n*(sc.e**2)*tau)/(sc.m_e) * ( 1.0/(1.0+np.power(2.0*sc.pi*freq*tau,2.0) )) #Beweis, dass es passt: https://sci-hub.st/10.1109/TASC.2012.2237515
            sigma2_t = (n*ns_n*(sc.e**2))/(sc.m_e*(2.0*sc.pi*freq) ) * ( 1.0 + nn_n/ns_n * ( np.power(2.0*sc.pi*freq*tau,2.0)) /( 1.0 + np.power(2.0*sc.pi*freq*tau,2.0) ) )
            sigma1.append(sigma1_t)
            sigma2.append(sigma2_t)

        return sigma1, sigma2

    def calc_Zs(self,freq,sigma1,sigma2):
        Zs = []
        #out = ''
        for i in range(0,len(sigma1)):
        #    out = out + str(freq) + ',' + str(sigma1[i]) + ',' + str(sigma2[i]) + '\n'
            Zs.append(csqrt( (1j*2*sc.pi*freq*sc.mu_0)/(sigma1[i] - 1j*sigma2[i])))
        #with open('C:\\Users\\finn-\\sigma.txt','a') as file:
        #    file.write(out)
        #    file.close()
        return Zs

    def export_csv(self,Zs_sweep,T_sweep,file):
        try:
            file.write('sep=,\n')
            file.write('Temperature[K],Rs[Ohm],Xs[Ohm]')
            i = 0
            for Zs in Zs_sweep:
                file.write('\n' + str(T_sweep[i]) + ',' + str(np.real(Zs)) + ',' + str(np.imag(Zs)))
                i = i + 1
            file.close()
        except:
            pass

    def export_for_cst(self,Zs_sweep,freqs,file):
        try:
            file.write('frequency[GHz]|Rs[Ohm]|Xs[Ohm]')
            i = 0
            for Zs in Zs_sweep:
                file.write('\n' + str(freqs[i]*1e-9) + ' ' + str(np.real(Zs)) + ' ' + str(np.imag(Zs)))
                i = i + 1
            file.close()
        except:
            pass

    def plot_results(self,params): #plot the calculation results according to calculation mode
        fig_zjj, axes_zjj = plt.subplots(nrows=1, ncols=1)
        fig_zjj.set_size_inches(7, 6)

        fig_sep, axes_sep = plt.subplots(nrows=1, ncols=2)
        fig_sep.set_size_inches(13, 6)

        fig_comb, axes_comb = plt.subplots(nrows=1, ncols=1)
        fig_comb.set_size_inches(7, 6)
        
        ax_rs = axes_sep[0]
        ax_xs = axes_sep[1]
        ax_comb = axes_comb

        ax_rs.ticklabel_format(useOffset=False)
        ax_xs.ticklabel_format(useOffset=False)
        ax_comb.ticklabel_format(useOffset=False)
        axes_zjj.ticklabel_format(useOffset=False)

        ax_rs.set_yscale('log')
        ax_xs.set_yscale('log')
        ax_comb.set_yscale('log')

        if(params['mode'] == 'freq'):

            ax_xs.set_xscale('log')
            ax_rs.set_xscale('log')
            ax_comb.set_xscale('log')
            axes_zjj.set_xscale('log')
            
            
            if all(values < 0 for values in np.imag(np.power(params['Zs'],-1))):
                print("Die Suzeptanz ist induktiv.")
                susc_type = 'ind.'
            else:
                print("Die Suzeptanz ist kapazitiv.")
                susc_type = 'cap.'

            #ax_xs.plot(params['freqs'],np.absolute(np.imag(np.power(params['Zs'],-1))),label='|Bs| (' + susc_type +')',color='b')
            #ax_rs.plot(params['freqs'],np.real(np.power(params['Zs'],-1)),label='Gs',color='r')
            #ax_comb.plot(params['freqs'],np.absolute(np.imag(np.power(params['Zs'],-1))),label='|Bs| (' + susc_type +')',color='b')
            #ax_comb.plot(params['freqs'],np.real(np.power(params['Zs'],-1)),label='Gs', color='r')

            ax_xs.plot(params['freqs'],np.imag(params['Zs']),label='Xs',color='b')
            ax_rs.plot(params['freqs'],np.real(params['Zs']),label='Rs',color='r')
            ax_comb.plot(params['freqs'],np.imag(params['Zs']),label='Xs',color='b')
            ax_comb.plot(params['freqs'],np.real(params['Zs']),label='Rs', color='r')
            axes_zjj.plot(params['freqs'],params['ZJJ'],label='Zjj',color='g')

            ax_rs.set_xlabel("Frequency [Hz]")
            ax_xs.set_xlabel("Frequency [Hz]")
            ax_comb.set_xlabel("Frequency [Hz]")
            axes_zjj.set_xlabel("Frequency [Hz]")
            
            #ax_rs.set_ylabel("Gs (S)")
            #ax_xs.set_ylabel("|Bs (S)|")
            #ax_comb.set_ylabel("Admittance (S)")

            ax_rs.set_ylabel("Rs (Ohm)")
            ax_xs.set_ylabel("Xs (Ohm)")
            ax_comb.set_ylabel("Impedance (Ohm)")
            axes_zjj.set_ylabel("Impedance (Ohm)")

        elif(params['mode'] == 'temp'):
            
            ax_xs.plot(params['T_sweep'],np.imag(params['Zs']),label='Xs',color='b')
            ax_rs.plot(params['T_sweep'],np.real(params['Zs']),label='Rs', color='r')
            ax_comb.plot(params['T_sweep'],np.imag(params['Zs']),label='Xs',color='b')
            ax_comb.plot(params['T_sweep'],np.real(params['Zs']),label='Rs', color='r')
            axes_zjj.plot(params['T_sweep'],params['ZJJ'],label='Zjj',color='g')

            ax_rs.set_xlabel("Temperature [K]")
            ax_xs.set_xlabel("Temperature [K]")
            ax_comb.set_xlabel("Temperature [K]")
            axes_zjj.set_xlabel("Temperature [K]")

            ax_rs.set_ylabel("Rs (Ohm)")
            ax_xs.set_ylabel("Xs (Ohm)")
            ax_comb.set_ylabel("Impedance (Ohm)")
            axes_zjj.set_ylabel("Impedance (Ohm)")

        fig_sep.tight_layout()
        fig_comb.tight_layout()
        
        ax_rs.legend()
        ax_xs.legend()
        ax_comb.legend()

        plt.show()

    def temp_sweep(self,sweep_params,calc_params): #Zs: temperature sweep
        sigma1,sigma2 = self.calc_sigma(sweep_params['freq'],sweep_params['T_sweep'],calc_params)
        Zs = self.calc_Zs(sweep_params['freq'],sigma1,sigma2)
        
        #temp correction doesnt work for the temperature sweep, because it's are only valid for t<<tc!
        #so thickness correction with lambda_0 only
        #Zs_temp_corrected = []
        #for i in range(0,len(Zs)):
        #    london_depth = self.london_depth_temp_correct(sweep_params['T_sweep'][i],calc_params)
        #    Zs_temp_corrected.append(Zs[i]*self.coth_thickness(15000*1e-9, london_depth))
        #Zs = Zs_temp_corrected
        
        Zs = np.asarray(Zs)*self.coth_thickness(calc_params['cond_thickness'], calc_params['london_depth_0'])
        Z_jj = self.calc_Z_jj(calc_params,sweep_params['T_sweep'])
        result = {'mode':'temp','ZJJ':Z_jj,'Zs':Zs,'T_sweep':sweep_params['T_sweep']}
        return result

    def coth_thickness(self,d,london_depth): #correction factor for conductor thickness
        return np.cosh(d/london_depth)/np.sinh(d/london_depth)

    def freq_sweep(self,sweep_params,calc_params): #Zs: frequency sweep
        Zs_sweep = []
        Zjj_sweep = []
        for freq in sweep_params['freqs']:
            sigma1,sigma2 = self.calc_sigma(freq,sweep_params['T_sweep'],calc_params)
            Zs = self.calc_Zs(freq,sigma1,sigma2)
            london_depth = self.london_depth_temp_correct(calc_params,sweep_params['T_sweep'][0])
            Zs = np.asarray(Zs)*self.coth_thickness(calc_params['cond_thickness'], london_depth)
            Z_jj = self.calc_Z_jj(calc_params,sweep_params['T_sweep'])
            Zs_sweep.append(Zs[0])
            Zjj_sweep.append(Z_jj[0])
        
        result = {'mode':'freq','ZJJ':Zjj_sweep,'Zs':Zs_sweep,'freqs':sweep_params['freqs']}
        return result

    def custom_temp_sweep(self): #wrapper function. not used by gui, serves as an example to use the main methods of this class

        #sweep_params = {'T_sweep':np.linspace(5,85.999,100),'freq':14.4e9}
        #calc_params = {'tc':90.0,'tau_tc':(7.5e-4)/(2*sc.pi*1.14e9),'beta':0.005,'alpha':5.5,'delta':0.5,'n':3e26}
        #sweep_params = {'T_sweep':np.linspace(10,93.49999,90),'freq':9.43e9}
        #calc_params = {'tc':93.5,'tau_tc':(0.004)/(2*sc.pi*9.43e9),'beta':0.2,'alpha':5.5,'delta':0.5,'n':3.0e26}

        sweep_params = {'T_sweep':np.linspace(3,81.99,100),'freq':1400e9}
        calc_params = {'tc':82.0,'tau_tc':(7.5e-4)/(2*sc.pi*1.14e9),'alpha':0.47, 'beta':0.005,'london_depth_0':150e-9,'cond_thickness': 150e-9}
        return self.temp_sweep(sweep_params,calc_params)

    def custom_freq_sweep(self): #wrapper function. not used by gui, serves as an example to use the main methods of this class
        #params Nuss
        #sweep_params = {'T_sweep':[77],'freqs':np.linspace(10e9,3000e9,300)}
        #calc_params = {'tc':90.0,'tau_tc':(7.5e-4)/(2*sc.pi*1.14e9),'beta':1.0,'alpha':5.5,'delta':0.5,'n':0.4e26}
        
        #MA: full range
        #sweep_params = {'T_sweep':[20],'freqs':np.linspace(10e9,3000e9,3000)}
        #calc_params = {'tc':86.0,'tau_tc':(7.5e-4)/(2*sc.pi*1.14e9),'beta':0.2,'alpha':5.5,'delta':0.5,'n':0.4e26}

        #MA: 10-60
        #sweep_params = {'T_sweep':[20],'freqs':np.linspace(10e9,60e9,100)}
        #calc_params = {'tc':86.0,'tau_tc':(7.5e-4)/(2*sc.pi*1.14e9),'beta':0.005,'alpha':5.5,'delta':0.5,'n':4.4e26}

        #MA: 700-800
        sweep_params = {'T_sweep':[22],'freqs':np.linspace(10e9,3000e9,4000)}
        calc_params = {'tc':82.0,'tau_tc':(7.5e-4)/(2*sc.pi*1.14e9),'alpha':0.47,'beta':0.005,'london_depth_0':150e-9,'cond_thickness':150e-9}

        #sweep_params = {'T_sweep':[20],'freqs':np.linspace(10e9,3000e9,300)}
        #calc_params = {'tc':90.0,'tau_tc':(7.5e-4)/(2*sc.pi*1.14e9),'beta':0.2,'alpha':5.5,'delta':0.5,'n':3e26}
        #print(np.imag(Zs)/(2*sc.pi*21e9)*1e13) #calculate L to verify Inductance with: https://sci-hub.st/10.1103/physrevb.49.9924
        return self.freq_sweep(sweep_params,calc_params)
    
    def calc_copper_surface_impedance(self,freqs): #not referenced in the masters thesis. calculates the surface impedance of copper for high frequencies
        sigma_0 = 5.96e7
        resistivity = 3e-3*1e-8 #https://www.copper.org/resources/properties/cryogenic/
        scattering_time = sc.m_e/(8.48e28*sc.e**2*resistivity)
        Zs_sweep = []
        for freq in freqs:
            sigma_r = sigma_0/(1-1j*2*sc.pi*freq*scattering_time)
            print(np.real(sigma_r))
            plasma_grenzfrequenz = csqrt(sigma_r/(sc.epsilon_0*scattering_time))
            Zs_sweep.append(csqrt( (-1j*2*sc.pi*freq*sc.mu_0*1)/(sigma_r-1j*2*sc.pi*freq*sc.epsilon_0) ))
        return Zs_sweep

    def copper_surface_impedance(self): #wrapper function for copper surface impedance calculation
        freqs = np.linspace(100e9,1000e9,200)
        Zs_sweep = self.calc_copper_surface_impedance(freqs)
        self.plot_results({'mode':'freq','Zs':Zs_sweep,'freqs':freqs})
        self.export_for_cst(Zs_sweep,freqs)

#use this class as follows:
#SurfImpSim = SurfImp_Algorithm()
#result = SurfImpSim.custom_freq_sweep()
#SurfImpSim.export_for_cst(result['Zs'],result['freqs'])
#SurfImpSim.export_csv(result['Zs'],result['T_sweep'] )
#SurfImpSim.plot_results(result)

#for other materials, adjust the parameters in the custom_freq_sweep function (calc_params). Refer to 10.1134/1.1326973 or to the thesis for an explanation on the parameters.

#for an approximation for the surface impedance of copper use
#SurfImpSim.copper_surface_impedance()

#Copper oxide planes in high-temperature superconductors can be considered thin
#conducting layers, with thickness c for YBa2Cu3O72δ corresponding to a sheet
#resistance ρab=1
#2c. Using this layer approximation, the IoffeRegel parameter kFl
#mentioned in Section III.A can be estimated from the expression
#https://sci-hub.st/10.1016/B978-0-12-409509-0.00002-0