#!/usr/bin/env python3
import sys
import numpy as np
import ROOT
sys.path.append('EFT2Obs/scripts')
from eftscaling import EFTScaling

# First we load the EFT scaling information from the json file
eft_scaling = EFTScaling.fromJSON("EFT2Obs/HiggsVBF_HiggsPt.json")

# Number of bins in the distribution
N = eft_scaling.nbins

# Now we want to estimate the covariance matrix of our hypothetical measurement
# First let's figure out how many SM Higgs events we expect to see, say in the Run 2
# dataset of 138 fb-1. We can get the SM prediction from RIVET via eft_scaling.sm_vals.
# This corresponds to the cross section per bin (in fb), divided by the number of events
# in each RIVET run (i.e. each individual job, not the sum of the total). So if we ran
# with `-e 20000`, we need to multiply by 20000 below. Change this value to whatever you
# used!
sm_evts = eft_scaling.sm_vals * 20000. * 138.

# Now let's assume we have 3x the number of background events for each signal event
# Feel free to try and refine this estimate!
bkg_evts = sm_evts * 3.

tot_evts = sm_evts + bkg_evts

# The relative stat uncertainty on the number of signal events:
stat_uncert = np.sqrt(tot_evts) / sm_evts

print("StatRel", stat_uncert)

# Let's also add some uncertainty on the number of background events
# E.g. we could add 5% of the background in each bin
bkg_syst = (bkg_evts * 0.05) / sm_evts # let's say 5% uncertainty on the background yield
print("SystRel", bkg_syst)

# Now define the covariance matrix for the statistics part, which will have elements
# only on the diagonal
cov_stat = np.identity(N) * np.power(stat_uncert, 2)
print(cov_stat)

# For the systematics part, let's assume the uncertainty is fully correlated between bins
# So we form the matrix as:
cov_syst = np.array([bkg_syst]).T * np.array([bkg_syst])

# Total covariance matrix is just the sum of the two:
cov = cov_syst + cov_stat

## Now we come to the fitting part
# It will be convenient to store objects inside a RooWorkspace
wsp = ROOT.RooWorkspace()

# We need two RooArgLists - one will contain objects representing the signal strengths
# as a function of the Wilson coefficients
mu_arglist = ROOT.RooArgList()
# The other is a list of constants to represent the "measured" values of the signal strengths
# In this case they will all be 1.0 - i.e. the SM
x_arglist = ROOT.RooArgList()

# This returns a list of parameter names, e.g. ["chw", "chdd", ...]
params = eft_scaling.parameters()

# We use the workspace factory syntax to create a RooRealVar for each parameter
# The range of -100,100 is a guess - you might need to fine-tune this!
for param in params:
    wsp.factory("{}[0, -100, 100]".format(param))

# We can print all the parameters that are in the workspace
wsp.allVars().Print("v")

# Now let's loop through each bin of the distribution, and build the scaling function
# as a RooFormulaVar
for i in range(N):
    # Each term starts with the SM
    expr = "1."
    for X in eft_scaling.terms:
        # Each entry in "terms" is an EFTTerm object (see EFTScaling.py)
        # this contains the list of parameters (one or two entries), 
        # the array of coefficient values (Ai or Bjk) and the uncertainties
        expr += " + " + ("%.4g*" % X.val[i]) + "*".join(X.params)
    wsp.factory("expr::scale_{}('{}',{})".format(i, expr, ','.join(params)))

    # Now create the measured value, just a constant
    wsp.factory("measured_{}[1.]".format(i))
    mu_arglist.add(wsp.function("scale_{}".format(i)))
    x_arglist.add(wsp.var("measured_{}".format(i)))
wsp.allFunctions().Print("v")

# Now we will build our PDF as a multi-variate Gaussian. We have to convert
# to ROOT format first
root_cov = ROOT.TMatrixDSym(N)
for i in range(N):
    for j in range(N):
        root_cov[i][j] = cov[i][j]

pdf = ROOT.RooMultiVarGaussian('pdf', '', mu_arglist, x_arglist, root_cov)

# We need to make a kind of fake "data" to form the likelihood, this is just
# a data set with a single point (corresponding to each measured bin = 1)
dat = ROOT.RooDataSet('global_obs', '', x_arglist)
dat.add(x_arglist)

# Now create the NLL function
nll = pdf.createNLL(dat)

## Now we can do some fitting! 

def RunFits(wsp, nll, params=list(), oneAtATime=True, doScans=True, outputName='test', scanRangeMultiplier=3):
    """Run fits and scans for the parameters of the model

    Args:
        wsp (RooWorkspace): input workspace
        nll (RooAbsReal): the nll function
        params (list, optional): list of parameter to fit. Defaults to list().
        oneAtATime (bool, optional): whether to float just one parameter at a time (or all of them). Defaults to True.
        doScans (bool, optional): perform and save scans of the profile likelihood. Defaults to True.
        outputName (str, optional): a string that will be appended to the output files. Defaults to 'test'.
        scanRangeMultiplier (int, optional): The function will try and guess a reasonable parameter range to scan - can be tuned with this multiplier. Defaults to 3.
    """
    print("Fitting parameters: {}, oneAtATime={}".format(params, oneAtATime))
    minim = ROOT.RooMinimizer(nll)
    minim.setEps(0.01)
    minim.setVerbose(False)

    # Set all the parameters floating and back to zero first
    for param in params:
        wsp.var(param).setVal(0.)
        wsp.var(param).setConstant(False)
    
    if not oneAtATime:
        # We should do the global fit, and run HESSE to get a good
        # initial estimate of the uncertainties
        minim.minimize('Minuit2','migrad')
        minim.hesse()
        print(">>> Fit result:")
        fitresult = minim.save()
        fitresult.Print()
        # Interesting to check the correlation matrix here
        print(">>> Correlation matrix:")
        fitresult.correlationMatrix().Print()
    
    # We can save a snapshot of the parameter states, and restore it later
    wsp.saveSnapshot("initial", wsp.allVars())

    for param in params:
        # restore the initial snapshot
        wsp.loadSnapshot("initial")
        if oneAtATime:
            # We need to set all the other parameters to constant
            for p2 in params:
                if p2 != param:
                    wsp.var(p2).setConstant(True)
            minim.setPrintLevel(-1)
            minim.minimize('Minuit2','migrad')
            minim.hesse()
            print(">> Fit result for {}:".format(param))
            wsp.var(param).Print()
    
        if doScans:
            minim.setPrintLevel(-1)

            var = wsp.var(param)
            print("Scanning {} = {:.3f} +/- {:.3f}".format(param, 
                var.getVal(), var.getError()))

            # Create an output ROOT file in the same format combine makes
            # this is just so we can use existing NLL plotting script
            fname = 'scan_{}_{}.root'.format(outputName, param)
            fout = ROOT.TFile(fname, 'RECREATE')
            tout = ROOT.TTree('limit', 'limit')

            a_r = np.array([var.getVal()], dtype='f')
            a_deltaNLL = np.array([0], dtype='f')
            a_quantileExpected = np.array([1], dtype='f')

            tout.Branch(param, a_r, '%s/f' % param)
            tout.Branch('deltaNLL', a_deltaNLL, 'deltaNLL/f')
            tout.Branch('quantileExpected', a_quantileExpected, 'quantileExpected/f')
        
            # Fill the initial (best-fit values)
            tout.Fill()
            nll0 = nll.getVal()

            # Now do a scan
            var.setConstant(True)
            npoints = 50
            r_min = var.getVal()-scanRangeMultiplier*var.getError()
            r_max = var.getVal()+scanRangeMultiplier*var.getError()
            width = (float(r_max) - float(r_min)) / float(npoints)
            r = r_min + 0.5 * width

            for p in range(1,npoints+1):
                var.setVal(r)
                a_r[0] = r
                minim.minimize('Minuit2', 'migrad')
                nllf = nll.getVal()
                print('%s: %s = %f; nll0 = %f; nll = %f, deltaNLL = %f' % (p, param, r, nll0, nllf,  nllf - nll0))
                a_deltaNLL[0] = nllf - nll0
                if a_deltaNLL[0] > 0.: 
                    tout.Fill()
                r += width
        
            tout.Write()
            fout.Close()


# Experiment with uncommenting the lines below!

# RunFits(wsp, nll, params=params, oneAtATime=False, doScans=True)
# RunFits(wsp, nll, params=params, oneAtATime=True, doScans=True, scanRangeMultiplier=1)
 
