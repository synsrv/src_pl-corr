
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl

import unittest, os
import numpy as np

from model.model import powerlaw_distribution

class TestPowerLawDistribution(unittest.TestCase):

    def test_method_and_generate_plot(self):

        np.random.seed(20191007)

        samples, xmin = 100000, 1
        bins = np.logspace(start=np.log10(xmin),
                           stop=np.log10(10**10),
                           num=100)

        fig, ax = pl.subplots()
        fig.set_size_inches(5.2,3)

        for alpha in [1.25,1.5,2.5,3.5]:

            ys = powerlaw_distribution(samples, alpha, xmin)
         
            counts, edges = np.histogram(ys, bins=bins,
                                         density=True)
            centers = (edges[:-1] + edges[1:])/2

            ax.plot(centers, counts)           
               

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_xlabel('t')
        ax.set_ylabel('f(t)')

        fig.savefig('tests/unit_tests/' +
                    'test_powerlaw_distribution_output.png',
                    dpi=300, bbox_inches='tight')




if __name__ == "__main__":
    unittest.main()



