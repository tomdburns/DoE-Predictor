#!/usr/bin/env python

"""
The is the code that runs the prediction for DoE PRT
based on user's inputted variable

VERSION 1.0
"""


import os
from Tkinter import *
import pickle as pkl
import numpy as np
from math import log10 as log
from math import exp
from sklearn import tree


RG1   = 8.314472      # Universal gas constant [m^3*Pa/(K*mol)]
RG2   = RG1 / 100000. # [m^3*bar/(K*mol)]
TEMP  = 298.15        # Kelvin
RT1   = RG2 * TEMP
RT2   = RG1 * TEMP
SPATH = '/'.join(os.path.realpath(__file__).split('/')[:-1])


def import_model():
    """Imports the data from the fitted model"""
    mod_file = open('{}/model_data/RandomForest.model'.format(SPATH), 'rb')
    model = pkl.load(mod_file)
    mod_file.close()
    return model


def import_offsets():
    """Imports the set of offsets"""
    offsets = {}
    data = open('{}/model_data/offset_vals.csv'.format(SPATH), 'r').readlines()[1:]
    for line in data:
        offsets[line.split(',')[0]] = float(line.split(',')[1])
    return offsets


def import_logs():
    """Imports the set of columns that need to be logged"""
    data = open('{}/model_data/logged.vars'.format(SPATH), 'rb')
    cols = pkl.load(data)
    data.close()
    return cols


def hoa2U(hoa):
    """Converts the heat of adsorption to internal energy"""
    if hoa > 0.0:
        hoa = hoa * -1.0
    hoa = 1000 * hoa
    return hoa + RT2


def first_window():
    """Opens the first window"""
    root = Tk()
    Label(root, text="MOF DoE Purity Recovery Target Predictor").grid(row=0, column=1)

    Label(root, text="K for strong carbon site").grid(row=1)
    Label(root, text="[1/bar]").grid(row=1, column=3)
    Label(root, text="K for weak carbon site").grid(row=2)
    Label(root, text="[1/bar]").grid(row=2, column=3)
    Label(root, text="Qsat for strong carbon site").grid(row=3)
    Label(root, text="[mmol/g]").grid(row=3, column=3)
    Label(root, text="Qsat for weak carbon site").grid(row=4)
    Label(root, text="[mmol/g]").grid(row=4, column=3)

    Label(root, text="K for strong nitrogen site").grid(row=6)
    Label(root, text="[1/bar]").grid(row=6, column=3)
    Label(root, text="K for weak nitrogen site").grid(row=7)
    Label(root, text="[1/bar]").grid(row=7, column=3)
    Label(root, text="Qsat for strong nitrogen site").grid(row=8)
    Label(root, text="[mmol/g]").grid(row=8, column=3)
    Label(root, text="Qsat for weak nitrogen site").grid(row=9)
    Label(root, text="[mmol/g]").grid(row=9, column=3)

    Label(root, text="CO2 heat of adsorption").grid(row=11)
    Label(root, text="[kJ/mol]").grid(row=11, column=3)
    Label(root, text="N2 heat of adsorption").grid(row=12)
    Label(root, text="[kJ/mol]").grid(row=12, column=3)

    Label(root, text="MOF Crystal Density").grid(row=14)
    Label(root, text="[kg/m^3]").grid(row=14, column=3)

    Label(root, text=" ").grid(row=5, column=0)
    Label(root, text=" ").grid(row=10, column=0)
    Label(root, text=" ").grid(row=13, column=0)

    e1 = Entry(root)
    e2 = Entry(root)
    e3 = Entry(root)
    e4 = Entry(root)
    e5 = Entry(root)
    e6 = Entry(root)
    e7 = Entry(root)
    e8 = Entry(root)
    e9 = Entry(root)
    e10 = Entry(root)
    e11 = Entry(root)

    e1.grid(row=1, column=1)
    e2.grid(row=2, column=1)
    e3.grid(row=3, column=1)
    e4.grid(row=4, column=1)
    e5.grid(row=6, column=1)
    e6.grid(row=7, column=1)
    e7.grid(row=8, column=1)
    e8.grid(row=9, column=1)
    e9.grid(row=11, column=1)
    e10.grid(row=12, column=1)
    e11.grid(row=14, column=1)

    Button(root, text='Quit', command=quit_program).grid(row=15, column=2, sticky=W, pady=4)
    Button(root, text="Enter", command=root.quit).grid(row=15, column=1, sticky=W, pady=4)

    mainloop()

    originals = [float(e1.get()), float(e2.get()), float(e3.get()),
                 float(e4.get()), float(e5.get()),
                 float(e6.get()), float(e7.get()), float(e8.get()),
                 float(e9.get()), float(e10.get()),
                 float(e11.get())]

    bc_1 = float(e1.get()) * RT1
    bc_2 = float(e2.get()) * RT1
    qc_1 = float(e3.get())
    qc_2 = float(e4.get())

    bn_1 = float(e5.get()) * RT1
    bn_2 = float(e6.get()) * RT1
    qn_1 = float(e7.get())
    qn_2 = float(e8.get())

    choa = float(e9.get())
    cU   = hoa2U(choa)
    nhoa = float(e10.get())
    nU   = hoa2U(nhoa)
    dens = float(e11.get()) * 0.75

    b0_c = bc_1 * exp(cU / RT2)
    d0_c = bc_2 * exp(cU / RT2)

    b0_n = bn_1 * exp(nU / RT2)
    d0_n = bn_2 * exp(nU / RT2)
    root.destroy()

    return {'b0_c': b0_c, 'd0_c': d0_c, 'q1_c': qc_1, 'q2_c': qc_2,
            'U1_c': cU, 'U2_c': cU,
            'b0_n': b0_n, 'd0_n': d0_n, 'q1_n': qn_1, 'q2_n': qn_2,
            'U1_n': nU, 'U2_n': nU, 'StructuredDensity': dens}, originals


def sub_window(params, result, originals):
    """Opens the first window"""
    if result == 0:
        banner = "MOF Will Not Pass DoE PRT"
    else:
        banner = "MOF Will Pass DoE PRT"

    root = Tk()
    Label(root, text="MOF DoE Purity Recovery Target Predictor").grid(row=0, column=1)

    Label(root, text="K for strong carbon site").grid(row=1)
    Label(root, text="[1/bar]").grid(row=1, column=3)
    Label(root, text="K for weak carbon site").grid(row=2)
    Label(root, text="[1/bar]").grid(row=2, column=3)
    Label(root, text="Qsat for strong carbon site").grid(row=3)
    Label(root, text="[mmol/g]").grid(row=3, column=3)
    Label(root, text="Qsat for weak carbon site").grid(row=4)
    Label(root, text="[mmol/g]").grid(row=4, column=3)

    Label(root, text="K for strong nitrogen site").grid(row=6)
    Label(root, text="[1/bar]").grid(row=6, column=3)
    Label(root, text="K for weak nitrogen site").grid(row=7)
    Label(root, text="[1/bar]").grid(row=7, column=3)
    Label(root, text="Qsat for strong nitrogen site").grid(row=8)
    Label(root, text="[mmol/g]").grid(row=8, column=3)
    Label(root, text="Qsat for weak nitrogen site").grid(row=9)
    Label(root, text="[mmol/g]").grid(row=9, column=3)

    Label(root, text="CO2 heat of adsorption").grid(row=11)
    Label(root, text="[kJ/mol]").grid(row=11, column=3)
    Label(root, text="N2 heat of adsorption").grid(row=12)
    Label(root, text="[kJ/mol]").grid(row=12, column=3)

    Label(root, text="MOF Crystal Density").grid(row=14)
    Label(root, text="[kg/m^3]").grid(row=14, column=3)
    Label(root, text=banner).grid(row=16, column=0)

    Label(root, text=" ").grid(row=5, column=0)
    Label(root, text=" ").grid(row=10, column=0)
    Label(root, text=" ").grid(row=13, column=0)

    e1 = Entry(root)
    e1.insert(10, originals[0])
    e2 = Entry(root)
    e2.insert(10, originals[1])
    e3 = Entry(root)
    e3.insert(10, originals[2])
    e4 = Entry(root)
    e4.insert(10, originals[3])
    e5 = Entry(root)
    e5.insert(10, originals[4])
    e6 = Entry(root)
    e6.insert(10, originals[5])
    e7 = Entry(root)
    e7.insert(10, originals[6])
    e8 = Entry(root)
    e8.insert(10, originals[7])
    e9 = Entry(root)
    e9.insert(10, originals[8])
    e10 = Entry(root)
    e10.insert(10, originals[9])
    e11 = Entry(root)
    e11.insert(10, originals[10])

    e1.grid(row=1, column=1)
    e2.grid(row=2, column=1)
    e3.grid(row=3, column=1)
    e4.grid(row=4, column=1)
    e5.grid(row=6, column=1)
    e6.grid(row=7, column=1)
    e7.grid(row=8, column=1)
    e8.grid(row=9, column=1)
    e9.grid(row=11, column=1)
    e10.grid(row=12, column=1)
    e11.grid(row=14, column=1)

    Button(root, text='Quit', command=quit_program).grid(row=15, column=2, sticky=W, pady=4)
    Button(root, text="Enter", command=root.quit).grid(row=15, column=1, sticky=W, pady=4)

    mainloop()

    originals = [float(e1.get()), float(e2.get()), float(e3.get()),
                 float(e4.get()), float(e5.get()),
                 float(e6.get()), float(e7.get()), float(e8.get()),
                 float(e9.get()), float(e10.get()),
                 float(e11.get())]

    bc_1 = float(e1.get()) * RT1
    bc_2 = float(e2.get()) * RT1
    qc_1 = float(e3.get())
    qc_2 = float(e4.get())

    bn_1 = float(e5.get()) * RT1
    bn_2 = float(e6.get()) * RT1
    qn_1 = float(e7.get())
    qn_2 = float(e8.get())

    choa = float(e9.get())
    cU   = hoa2U(choa)
    nhoa = float(e10.get())
    nU   = hoa2U(nhoa)
    dens = float(e11.get()) * 0.75

    b0_c = bc_1 * exp(cU / RT2)
    d0_c = bc_2 * exp(cU / RT2)

    b0_n = bn_1 * exp(nU / RT2)
    d0_n = bn_2 * exp(nU / RT2)
    root.destroy()

    return {'b0_c': b0_c, 'd0_c': d0_c, 'q1_c': qc_1, 'q2_c': qc_2,
            'U1_c': cU, 'U2_c': cU,
            'b0_n': b0_n, 'd0_n': d0_n, 'q1_n': qn_1, 'q2_n': qn_2,
            'U1_n': nU, 'U2_n': nU, 'StructuredDensity': dens}, originals


def quit_program():
    """Exits the program"""
    exit()


def make_descriptors(order, params, logs, offsets):
    """Sets up the descriptor for the model"""
    omit = 'idx,MOF,distances'.split(',')
    values = []
    for col in order:
        if col in omit:
            continue
        val = params[col]
        if col in offsets:
            val += (10 ** offsets[col])
        if col in logs:
            try:
                val = log(val)
            except ValueError:
                val = log(val + 10 ** -20)
        values.append(val)
    return np.array([values])


def run_model(descr, models):
    """Actually runs the model and returns the result"""
    passed, failed = 0, 0
    for model in models:
        val = model.predict(descr)[0]
        if val == 1:
            passed += 1
        else:
            failed += 1
    if passed > failed:
        return 1
    else:
        return 0


def main():
    """Main Execution"""
    running = True
    model   = import_model()
    offsets = import_offsets()
    logs    = import_logs()

    parameters, originals = first_window()
    descriptor = make_descriptors(model['Order'], parameters, logs, offsets)

    while running:
        result = run_model(descriptor, model['Models'])

        parameters, originals = sub_window(parameters, result, originals)
        descriptor = make_descriptors(model['Order'], parameters, logs, offsets)


if __name__ in '__main__':
    main()
