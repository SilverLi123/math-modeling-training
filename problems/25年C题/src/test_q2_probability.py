"""Independent numerical and data-contract checks for Q2."""
import unittest
import numpy as np
from scipy.integrate import quad
from scipy.special import expit
from scipy.optimize._numdiff import approx_derivative
from q2_probability import prepare, normal_rule, objective, predict, FEATURES


class ProbabilityTests(unittest.TestCase):
    def test_gradient_against_finite_difference(self):
        x=np.array([[1,-1,.2],[1,.3,-.5],[1,.8,1.],[1,1.1,-1.]])
        y=np.array([0,1,1,0.])
        groups=np.array([0,0,1,1])
        nodes,w=normal_rule(80)
        theta=np.array([.4,.7,-.3,.2])
        _,grad=objective(theta,x,y,groups,nodes,w)
        numeric=approx_derivative(lambda t:objective(t,x,y,groups,nodes,w)[0],theta).ravel()
        np.testing.assert_allclose(grad,numeric,rtol=1e-5,atol=1e-7)

    def test_group_likelihood_against_quad(self):
        x=np.array([[1.,0,0],[1,0,0]])
        y=np.array([1.,0])
        nodes,w=normal_rule(160)
        theta=np.array([.7,0,0,np.log(1.3)])
        nll,_=objective(theta,x,y,np.array([0,0]),nodes,w)
        exact=quad(lambda u:expit(.7+1.3*u)*expit(-.7-1.3*u)*np.exp(-u*u/2)/np.sqrt(2*np.pi),-10,10)[0]
        self.assertAlmostEqual(nll,-np.log(exact),places=9)

    def test_marginal_probability_not_zero_random_effect(self):
        nodes,w=normal_rule(160)
        p=float(expit(2+2*nodes)@np.exp(w))
        exact=quad(lambda u:expit(2+2*u)*np.exp(-u*u/2)/np.sqrt(2*np.pi),-10,10)[0]
        self.assertAlmostEqual(p,exact,places=9)
        self.assertGreater(abs(p-expit(2)),.05)

    def test_bmi_is_frozen_and_labels_unchanged(self):
        data,_=prepare()
        self.assertEqual(len(data),1082)
        self.assertEqual(data.attained.sum(),937)
        self.assertEqual(data.groupby('subject_id').baseline_bmi.nunique().max(),1)
        first=data.sort_values(['subject_id','test_date_iso','gestational_week','sample_id']).drop_duplicates('subject_id')
        np.testing.assert_allclose(first.bmi,first.baseline_bmi)
        self.assertEqual(FEATURES,['gestational_week','baseline_bmi'])


if __name__=='__main__':
    unittest.main()
