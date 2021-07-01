import ray
import warnings
from itertools import compress

import numpy as np
from scipy.special import logsumexp

from .chain import Chain, DAChain
from .proposal import *
from .utils import *


class ParallelChain:
    def __init__(self, link_factory, proposal, n_chains=2, initial_parameters=None):
        
        self.link_factory = link_factory
        self.proposal = proposal
        
        self.n_chains = n_chains
        self.initial_parameters = initial_parameters
        
        if self.initial_parameters is not None:
            if type(self.initial_parameters) == list:
                assert len(self.initial_parameters) == self.n_chains, 'If list of initial parameters is provided, it must have length n_chains'
            else:
                raise TypeError('Initial parameters must be a list')
        else:
            self.initial_parameters = list(self.link_factory.prior.rvs(self.n_chains))
        
        ray.init(ignore_reinit_error=True)
        
        self.remote_chains = [RemoteChain.remote(self.link_factory, 
                                                 self.proposal, 
                                                 initial_parameters) for initial_parameters in self.initial_parameters]
        
    def sample(self, iterations, progressbar=True):
        
        self.processes = [chain.sample.remote(iterations) for chain in self.remote_chains]
        self.chains = [ray.get(process) for process in self.processes]
        

class ParallelDAChain(ParallelChain):
    def __init__(self, link_factory_coarse, link_factory_fine, proposal, subsampling_rate=1, n_chains=2, initial_parameters=None, adaptive_error_model=None, R=None):
        
        # internalise link factories and the proposal
        self.link_factory_coarse = link_factory_coarse
        self.link_factory_fine = link_factory_fine
        self.proposal = proposal
        self.subsampling_rate = subsampling_rate
        
        self.n_chains = n_chains
        self.initial_parameters = initial_parameters
        
        self.adaptive_error_model = adaptive_error_model
        self.R = R
        
        if self.initial_parameters is not None:
            if type(self.initial_parameters) == list:
                assert len(self.initial_parameters) == self.n_chains, 'If list of initial parameters is provided, it must have length n_chains'
            else:
                raise TypeError('Initial parameters must be a list')
        else:
            self.initial_parameters = list(self.link_factory_coarse.prior.rvs(self.n_chains))
        
        ray.init(ignore_reinit_error=True)
        
        self.remote_chains = [RemoteDAChain.remote(self.link_factory_coarse, 
                                                   self.link_factory_fine, 
                                                   self.proposal, 
                                                   self.subsampling_rate, 
                                                   initial_parameters, 
                                                   self.adaptive_error_model, self.R) for initial_parameters in self.initial_parameters]

class FetchingDAChain:
    def __init__(self, link_factory_coarse, link_factory_fine, proposal, subsampling_rate=1, fetching_rate=1, initial_parameters=None):
        
        # if the proposal is pCN, Check if the proposal covariance is equal 
        # to the prior covariance and if the prior is zero mean.
        if isinstance(proposal, CrankNicolson) and not isinstance(link_factory_coarse.prior, stats._multivariate.multivariate_normal_frozen):
            raise TypeError('Prior must be of type scipy.stats.multivariate_normal for pCN proposal')
        
        # internalise link factories and the proposal
        self.link_factory_coarse = link_factory_coarse
        self.link_factory_fine = link_factory_fine
        self.proposal = proposal
        self.subsampling_rate = subsampling_rate
        self.fetching_rate = fetching_rate
        
        # set up lists to hold coarse and fine links, as well as acceptance
        # accounting
        self.chain_coarse = []
        self.accepted_coarse = []
        self.is_coarse = []
        
        self.chain_fine = []
        self.accepted_fine = []
        self.perfect_fetch = []
                
        # if the initial parameters are given, use them. otherwise,
        # draw a random sample from the prior.
        if initial_parameters is not None:
            self.initial_parameters = initial_parameters
        else:
            self.initial_parameters = self.link_factory_coarse.prior.rvs()
            
        # setup the proposal
        self.proposal.setup_proposal(parameters=self.initial_parameters, link_factory=self.link_factory_coarse)
        
        ray.init(ignore_reinit_error=True)
            
        self.coarse_workers = [RemoteSubchainFactory.remote(self.link_factory_coarse, self.subsampling_rate, self.fetching_rate) for i in range(self.fetching_rate+1)]
        self.fine_workers = [RemoteLinkFactory.remote(self.link_factory_fine) for i in range(self.fetching_rate)]
        
        initial_coarse_link = self.link_factory_coarse.create_link(self.initial_parameters)
            
        coarse_process = self.coarse_workers[0].run.remote(initial_coarse_link, self.proposal)
        fine_process = self.fine_workers[0].create_link.remote(self.initial_parameters)
        
        self.coarse_section = ray.get(coarse_process)
        self.chain_fine.append(ray.get(fine_process))
        self.accepted_fine.append(True)
        
    def sample(self, iterations):
        
        while len(self.chain_fine) < iterations:
            
            print('Progress: {0}/{1}, {2:.2f}%'.format(len(self.chain_fine), iterations, len(self.chain_fine)/iterations*100), end='\r')
            
            nodes = [link for link, is_coarse in zip(self.coarse_section['chain'], self.coarse_section['is_coarse']) if not is_coarse]
            
            coarse_process = [coarse_worker.run.remote(node, self.proposal) for coarse_worker, node in zip(self.coarse_workers, nodes)]
            fine_process = [fine_worker.create_link.remote(node.parameters) for fine_worker, node in zip(self.fine_workers, nodes[1:])]
            coarse_sections = ray.get(coarse_process); fine_links = ray.get(fine_process)
            
            self.perfect_fetch.append(True)
            
            for j in range(self.fetching_rate):
                
                alpha = np.exp(fine_links[j].posterior - self.chain_fine[-1].posterior + nodes[j].posterior - nodes[j+1].posterior)
                 
                if np.random.random() < alpha:
                    self.chain_fine.append(fine_links[j])
                    self.accepted_fine.append(True)
                else:
                    self.chain_fine.append(self.chain_fine[-1])
                    self.accepted_fine.append(False)
                    self.perfect_fetch[-1] = False
                    break
                    
            if self.perfect_fetch[-1]:
                fine_length = self.fetching_rate
            else:
                fine_length = j
            
            coarse_length = (j+1)*(self.subsampling_rate+1)
            self.chain_coarse.extend(self.coarse_section['chain'][:coarse_length])
            self.accepted_coarse.extend(self.coarse_section['accepted'][:coarse_length])
            self.is_coarse.extend(self.coarse_section['is_coarse'][:coarse_length])
            
            for j in range(coarse_length):
                if self.is_coarse[-coarse_length+j]:
                    self.proposal.adapt(parameters=self.chain_coarse[-coarse_length+j].parameters, 
                                        jumping_distance=self.chain_coarse[-coarse_length+j].parameters-self.chain_coarse[-coarse_length+j-1].parameters, 
                                        accepted=list(compress(self.accepted_coarse[:-coarse_length+j], self.is_coarse[:-coarse_length+j])))
                
            self.coarse_section = coarse_sections[fine_length]

class MultipleTry:
    
    '''
    Multiple-Try proposal, which will take any other TinyDA proposal
    as a kernel. The parameter k sets the number of tries.
    '''
    
    is_symmetric = True
    
    def __init__(self, kernel, k):
        
        # set the kernel
        self.kernel = kernel
        
        # set the number of tries per proposal.
        self.k = k
        
        if self.kernel.adaptive:
            warnings.warn(' Using global adaptive scaling with MultipleTry proposal can be unstable.\n')
            
        ray.init(ignore_reinit_error=True)
        
    def setup_proposal(self, **kwargs):
        
        # pass the kwargs to the kernel.
        self.kernel.setup_proposal(**kwargs)
        
        # initialise the link factories.
        self.link_factories = [RemoteLinkFactory.remote(kwargs['link_factory']) for i in range(self.k)]
        
    def adapt(self, **kwargs):
        
        # this method is not adaptive in its own, but its kernel might be.
        self.kernel.adapt(**kwargs)
        
    def make_proposal(self, link):
        
        # create proposals. this is fast so no paralellised.
        proposals = [self.kernel.make_proposal(link) for i in range(self.k)]
        
        # get the links in parallel.
        proposal_processes = [link_factory.create_link.remote(proposal) for proposal, link_factory in zip(proposals, self.link_factories)]
        self.proposal_links = [ray.get(proposal_process) for proposal_process in proposal_processes]
        
        # if kernel is symmetric, use MTM(II), otherwise use MTM(I).
        if self.kernel.is_symmetric:
            q_x_y = np.zeros(self.k)
        else:
            q_x_y = np.array([self.kernel.get_q(link, proposal_link) for proposal_link in self.proposal_links])
        
        # get the unnormalised weights.
        self.proposal_weights = np.array([link.posterior+q for link, q in zip(self.proposal_links, q_x_y)])
        self.proposal_weights[np.isnan(self.proposal_weights)] = -np.inf 
        
        # if all posteriors are -Inf, return a random onw.
        if np.isinf(self.proposal_weights).all():
            return np.random.choice(self.proposal_links).parameters
        
        # otherwise, return a random one according to the weights.
        else:            
            return np.random.choice(self.proposal_links, p=np.exp(self.proposal_weights - logsumexp(self.proposal_weights))).parameters
    
    def get_acceptance(self, proposal_link, previous_link):
        
        # check if the proposal makes sense, if not return 0.
        if np.isnan(proposal_link.posterior) or np.isinf(self.proposal_weights).all():
            return 0
        
        else:
            
            # create reference proposals.this is fast so no paralellised.
            references = [self.kernel.make_proposal(proposal_link) for i in range(self.k-1)]
            
            # get the links in parallel.
            reference_processes = [link_factory.create_link.remote(reference) for reference, link_factory in zip(references, self.link_factories)]
            self.reference_links = [ray.get(reference_process) for reference_process in reference_processes]
            
            # if kernel is symmetric, use MTM(II), otherwise use MTM(I).
            if self.kernel.is_symmetric:
                q_y_x = np.zeros(self.k)
            else:
                q_y_x = np.array([self.kernel.get_q(proposal_link, reference_link) for reference_link in self.reference_links])
            
            # get the unnormalised weights.
            self.reference_weights = np.array([link.posterior+q for link, q in zip(self.reference_links, q_y_x)])
            self.reference_weights[np.isnan(self.reference_weights)] = -np.inf 
            
            # get the acceptance probability.
            return np.exp(logsumexp(self.proposal_weights) - logsumexp(self.reference_weights))

@ray.remote
class RemoteChain(Chain):
    def sample(self, iterations, progressbar=False):
        super().sample(iterations, progressbar)
        
        return self.chain
        
@ray.remote
class RemoteDAChain(DAChain):
    def sample(self, iterations, progressbar=False):
        super().sample(iterations, progressbar)
        
        return self.chain_fine

@ray.remote
class RemoteSubchainFactory:
    
    def __init__(self, link_factory, subsampling_rate, fetching_rate):
        
        self.link_factory = link_factory
        self.subsampling_rate = subsampling_rate
        self.fetching_rate = fetching_rate
        
    def run(self, initial_link, proposal_kernel):
        
        chain = [initial_link]
        accepted = [True]
        is_coarse = [False]
        
        for i in range(self.fetching_rate):
        
            for j in range(self.subsampling_rate):
                
                # draw a new proposal, given the previous parameters.
                proposal = proposal_kernel.make_proposal(chain[-1])
                
                # create a link from that proposal.
                proposal_link = self.link_factory.create_link(proposal)
                
                # compute the acceptance probability, which is unique to
                # the proposal.
                alpha = proposal_kernel.get_acceptance(proposal_link, chain[-1])
                
                # perform Metropolis adjustment.
                if np.random.random() < alpha:
                    chain.append(proposal_link)
                    accepted.append(True)
                    is_coarse.append(True)
                else:
                    chain.append(chain[-1])
                    accepted.append(False)
                    is_coarse.append(True)
        
            chain.append(chain[-1])
            accepted.append(True)
            is_coarse.append(False)
            
        return {'chain': chain, 'accepted': accepted, 'is_coarse': is_coarse}

@ray.remote
class RemoteLinkFactory:
    def __init__(self, link_factory):
        self.link_factory = link_factory
        
    def create_link(self, parameters):
        return self.link_factory.create_link(parameters)