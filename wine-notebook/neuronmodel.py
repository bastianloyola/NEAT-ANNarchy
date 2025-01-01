
from ANNarchy import *

IZHIKEVICH = Neuron(  #I = 20
    parameters="""
        a = 0.02 : population
        b = 0.2 : population
        c = -65.0 : population
        d = 8.0 : population
        I = 0.0
        tau_I = 10.0 : population
    """,
    equations="""
        dv/dt = 0.04*v*v + 5*v + 140 - u + I + g_exc - g_inh : init=-65
        tau_I * dg_exc/dt = -g_exc
        tau_I * dg_inh/dt = -g_inh
        du/dt = a*(b*v - u) : init=-14.0
    """,
    spike="v >= 30.0",
    reset="v = c; u += d"
)
