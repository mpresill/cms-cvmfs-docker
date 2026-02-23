// -*- C++ -*-
#include "Rivet/Analysis.hh"
#include "Rivet/Math/Vector4.hh"
#include "Rivet/Projections/DressedLeptons.hh"
#include "Rivet/Projections/FastJets.hh"
#include "Rivet/Projections/FinalState.hh"
#include "Rivet/Projections/MissingMomentum.hh"
#include "Rivet/Projections/PromptFinalState.hh"

namespace Rivet {

class HiggsVBF : public Analysis {
public:
  /// Constructor
  DEFAULT_RIVET_ANALYSIS_CTOR(HiggsVBF);

  /// @name Analysis methods
  /// @{

  /// Book histograms and initialise projections before the run
  void init() {
    // Here we define the "projections" we are interested in

    // All final-state particles
    const FinalState fs;

    // Final-state particles within an acceptance of |eta| < 4.7
    // This matches the CMS acceptance for jet clustering
    const FinalState fsjet4p7(Cuts::abseta < 4.7);

    // The final-state particles declared above are clustered using FastJet with
    // the anti-kT algorithm and a jet-radius parameter 0.4
    // muons and neutrinos are excluded from the clustering
    FastJets jetfs(fsjet4p7, FastJets::ANTIKT, 0.4);
    // We declare a projection object, giving it a string name that we can use
    // to retrieve the result when analysing the events
    declare(jetfs, "jets");

    // FinalState of prompt photons and bare muons and electrons in the event
    PromptFinalState photons(Cuts::abspid == PID::PHOTON);
    PromptFinalState bare_leps(Cuts::abspid == PID::MUON ||
                               Cuts::abspid == PID::ELECTRON);
    bare_leps.acceptTauDecays(false);

    // Dress the prompt bare leptons with prompt photons within dR < 0.1,
    // and apply some fiducial cuts on the dressed leptons
    Cut lepton_cuts = Cuts::abseta < 2.5 && Cuts::pT > 20 * GeV;
    DressedLeptons dressed_leps(photons, bare_leps, 0.1, lepton_cuts);
    declare(dressed_leps, "leptons");

    // Missing momentum
    declare(MissingMomentum(fs), "MET");

    // Book histograms
    book(_h_HiggsPt, "HiggsPt", 10, 0, 500);
  }

  /// Perform the per-event analysis
  void analyze(const Event &event) {

    // Retrieve leptons, sorted by pT
    Particles leptons = apply<DressedLeptons>(event, "leptons").particlesByPt();

    // Reject events that do not contain two leptons
    if (!(leptons.size() == 2)) {
      return;
    }

    // Retrieve clustered jets, sorted by pT, with a minimum pT cut
    Jets jets = apply<FastJets>(event, "jets").jetsByPt(Cuts::pT > 30 * GeV);

    // Remove all jets within dR < 0.4 of a dressed lepton
    idiscardIfAnyDeltaRLess(jets, leptons, 0.4);

    // Apply a njets >= 2 cut
    if (jets.size() < 2) {
      return;
    }

    // Get the MET 4-vector
    FourMomentum met = applyProjection<MissingMomentum>(event, "MET").missingMomentum();

    // Calculate the Higgs candidate 4-vector as the sum of the MET and the two leptons
    FourMomentum Higgs = met + leptons[0].momentum() + leptons[1].momentum();


    // Fill histograms
    _h_HiggsPt->fill(Higgs.pT());
  }

  /// Normalise histograms etc., after the run
  void finalize() {
    double norm = (sumOfWeights() != 0)
                      ? crossSection() / femtobarn / sumOfWeights()
                      : 1.0;
    // Here we scale the histograms with a factor that makes the values equivalent to
    // cross sections in femtobarns
    scale(_h_HiggsPt, norm);
  }

private:
  // Define histogram pointers here
  Histo1DPtr _h_HiggsPt;
};

DECLARE_RIVET_PLUGIN(HiggsVBF);
} // namespace Rivet
