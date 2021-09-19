# Surface impedance calculator for high temperature superconductors

Based on 10.1134/1.1326973, this application is able to calculate the surface impedance for a hts by providing the london penetration depth at T = 0, the critical temperature Tc, the mean free time for the electrons at T = 0 and a couple of fitting parameters to match experimental data for conductivity, temperature dependant mean free time and superconducting electron density. The parameters are adjusted for YBCO though the model shows good agreement with other hts as well.

Two modes are available for calculation: frequency sweep and temperature sweep. The frequency sweep also allows to save the results to import them directly to CST Microwave Studio. A surface impedance material can then be used to accurately model the hts in simulations. Keep in mind that the frequency domain solver requires the option "Fit as in time domain" to be set to work accurately.
