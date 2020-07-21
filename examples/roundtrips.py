import demes

g1 = demes.load("gutenkunst_ooa.yml")
demes.dump(g1, "gutenkunst_ooa_output.yml")

g2 = demes.load("browning_america.yml")
demes.dump(g2, "browning_america_output.yml")

# an example with overlapping ancestors/descendants, mass migration event, symmetric migration, and asymmetric migration
g3 = demes.load("offshoots.yml")
demes.dump(g3, "offshoots_output.yml")
