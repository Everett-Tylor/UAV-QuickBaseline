import unittest
import torch
from dino_train import bare_focus_loss
from dino_pseudo import filter_labels


class BareFocusTests(unittest.TestCase):
    def test_high_confidence_bare_pixel_requires_more_agreement(self):
        probs=torch.zeros(1,9,1,2)
        probs[:,5]=.97
        probs[:,1]=.03
        votes=torch.full((8,1,1,2),5)
        votes[0,0,0,0]=1
        label,keep=filter_labels(probs,votes)
        self.assertEqual(label.tolist(),[[[0,0]]])
        probs[:,5]=.99
        label,keep=filter_labels(probs,votes)
        self.assertEqual(label.tolist(),[[[5,5]]])

    def test_background_false_bare_probability_has_gradient(self):
        logits=torch.zeros(1,9,2,2,requires_grad=True)
        target=torch.tensor([[[1,1],[5,0]]])
        loss=bare_focus_loss(logits,target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(logits.grad[0,5,0,0]),0.)
        self.assertEqual(float(logits.grad[0,:,1,1].abs().sum()),0.)


if __name__=='__main__':unittest.main(verbosity=2)
