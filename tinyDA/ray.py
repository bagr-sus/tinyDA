from math import log
import ray
import warnings

from itertools import compress
import numpy as np
from scipy.special import logsumexp

from .chain import Chain, DAChain, MLDAChain
from .proposal import *


STUCK_LIKELIHOOD_THRESHOLD = 1.5
STUCK_LIKELIHOOD_OFFSET = -80

PROGRESSION_CHECK_OFFSET = 500
PROGRESSION_LIKELIHOOD_THRESHOLD = 0.95

STUCK_CHECKING_PERIOD = 20 * 500
STUCK_CHECKING_START = 20 * 2000

class ParallelChain:

    """ParallelChain creates n_chains instances of tinyDA.Chain and runs the
    chains in parallel. It is initialsed with a Posterior (which holds the
    model and the distributions, and returns Links), and a proposal (transition
    kernel).

    Attributes
    ----------
    posterior : tinyDA.Posterior
        A posterior responsible for communation between prior, likelihood
        and model. It also generates instances of tinyDA.Link (sample objects).
    proposal : tinyDA.Proposal
        Transition kernel for MCMC proposals.
    n_chains : int
        Number of parallel chains.
    initial_parameters : list
        Starting points for the MCMC samplers
    remote_chains : list
        List of Ray actors, each running an independent MCMC sampler.

    Methods
    -------
    sample(iterations)
        Runs the MCMC for the specified number of iterations.
    """

    def __init__(self, posterior, proposal, n_chains=2, initial_parameters=None):
        """
        Parameters
        ----------
        posterior : tinyDA.Posterior
            A posterior responsible for communation between prior, likelihood
            and model. It also generates instances of tinyDA.Link (sample objects).
        proposal : tinyDA.Proposal
            Transition kernel for MCMC proposals.
        n_chains : int, optional
            Number of independent MCMC samplers. Default is 2.
        initial_parameters : list, optional
            Starting points for the MCMC samplers, default is None (random
            draws from prior).
        """

        # internalise the posterior and proposal.
        self.posterior = posterior
        self.proposal = proposal

        # set the number of parallel chains and initial parameters.
        self.n_chains = n_chains

        # set the initial parameters.
        self.initial_parameters = initial_parameters

        # initialise Ray.
        #ray.init(ignore_reinit_error=True)
        if not ray.is_initialized():
            ray.init(address="auto")

        # set up the parallel chains as Ray actors.
        self.remote_chains = [
            RemoteChain.remote(
                self.posterior, self.proposal[i], self.initial_parameters[i]
            )
            for i in range(self.n_chains)
        ]

    def sample(self, iterations, progressbar=False):
        """
        Parameters
        ----------
        iterations : int
            Number of MCMC samples to generate.
        progressbar : bool, optional
            Whether to draw a progressbar, default is False, since Ray
            and tqdm do not play very well together.
        """

        # initialise sampling on the chains and fetch the results.
        processes = [
            chain.sample.remote(iterations, progressbar) for chain in self.remote_chains
        ]
        self.chains = ray.get(processes)


class ParallelDAChain(ParallelChain):
    def __init__(
        self,
        posterior_coarse,
        posterior_fine,
        proposal,
        subchain_length=1,
        randomize_subchain_length=False,
        n_chains=2,
        initial_parameters=None,
        adaptive_error_model=None,
        store_coarse_chain=True,
    ):
        # internalise posteriors, proposal and subchain length.
        self.posterior_coarse = posterior_coarse
        self.posterior_fine = posterior_fine
        self.proposal = proposal
        self.subchain_length = subchain_length

        # set the number of parallel chains and initial parameters.
        self.n_chains = n_chains

        # set the initial parameters.
        self.initial_parameters = initial_parameters

        # set whether to randomize subchain length
        self.randomize_subchain_length = randomize_subchain_length

        # set the adaptive error model.
        self.adaptive_error_model = adaptive_error_model

        # whether to store the coarse chain.
        self.store_coarse_chain = store_coarse_chain

        # initialise Ray.
        #ray.init(ignore_reinit_error=True)
        if not ray.is_initialized():
            ray.init(address="auto")

        # set up the parallel DA chains as Ray actors.
        self.remote_chains = [
            RemoteDAChain.remote(
                self.posterior_coarse,
                self.posterior_fine,
                self.proposal[i],
                self.subchain_length,
                self.randomize_subchain_length,
                self.initial_parameters[i],
                self.adaptive_error_model,
                self.store_coarse_chain,
            )
            for i in range(self.n_chains)
        ]


class ParallelMLDAChain(ParallelChain):
    def __init__(
        self,
        posteriors,
        proposal,
        subchain_lengths=None,
        n_chains=2,
        initial_parameters=None,
        adaptive_error_model=None,
        store_coarse_chain=True,
    ):
        # internalise posteriors, proposal and subchain length.
        self.posteriors = posteriors
        self.proposal = proposal
        self.subchain_lengths = subchain_lengths

        # set the number of parallel chains and initial parameters.
        self.n_chains = n_chains

        # set the initial parameters.
        self.initial_parameters = initial_parameters

        # set the adaptive error model
        self.adaptive_error_model = adaptive_error_model

        # whether to store the coarse chain.
        self.store_coarse_chain = store_coarse_chain

        # initialise Ray.
        #ray.init(ignore_reinit_error=True)
        if not ray.is_initialized():
            ray.init(address="auto")

        # set up the parallel DA chains as Ray actors.
        self.remote_chains = [
            RemoteMLDAChain.remote(
                self.posteriors,
                self.proposal[i],
                self.subchain_lengths,
                self.initial_parameters[i],
                self.adaptive_error_model,
                self.store_coarse_chain,
            )
            for i in range(self.n_chains)
        ]


@ray.remote
class RemoteChain(Chain):
    def sample(self, iterations, progressbar):
        super().sample(iterations, progressbar)
        return self


@ray.remote
class RemoteDAChain(DAChain):
    def sample(self, iterations, progressbar):
        super().sample(iterations, progressbar)
        return self


@ray.remote
class RemoteMLDAChain(MLDAChain):
    def sample(self, iterations, progressbar):
        super().sample(iterations, progressbar)
        return self


class MultipleTry(Proposal):

    """Multiple-Try proposal (Liu et al. 2000), which will take any other
    TinyDA proposal as a kernel. If the kernel is symmetric, it uses MTM(II),
    otherwise it uses MTM(I). The parameter k sets the number of tries.

    Attributes
    ----------
    kernel : tinyDA.Proposal
        The kernel of the Multiple-Try proposal (another proposal).
    k : int
        Number of mutiple tries.

    Methods
    ----------
    setup_proposal(**kwargs)
        Initialises the kernel, and the remote Posteriors.
    adapt(**kwargs)
        Adapts the kernel.
    make_proposal(link)
        Generates a Multiple Try proposal, using the kernel.
    get_acceptance(proposal_link, previous_link)
        Computes the acceptance probability given a proposal link and the
        previous link.
    """

    is_symmetric = True

    def __init__(self, kernel, k):
        """
        Parameters
        ----------
        kernel : tinyDA.Proposal
            The kernel of the Multiple-Try proposal (another proposal)
        k : int
            Number of mutiple tries.
        """

        # set the kernel
        self.kernel = kernel

        # set the number of tries per proposal.
        self.k = k

        if self.kernel.adaptive:
            warnings.warn(
                " Using global adaptive scaling with MultipleTry proposal can be unstable.\n"
            )

        #ray.init(ignore_reinit_error=True)
        if not ray.is_initialized():
            ray.init(address="auto")

    def setup_proposal(self, **kwargs):
        # pass the kwargs to the kernel.
        self.kernel.setup_proposal(**kwargs)

        # initialise the posteriors.
        self.posteriors = [
            RemotePosterior.remote(kwargs["posterior"]) for i in range(self.k)
        ]

    def adapt(self, **kwargs):
        # this method is not adaptive in its own, but its kernel might be.
        self.kernel.adapt(**kwargs)

    def make_proposal(self, link):
        # create proposals. this is fast so no paralellised.
        proposals = [self.kernel.make_proposal(link) for i in range(self.k)]

        # get the links in parallel.
        proposal_processes = [
            posterior.create_link.remote(proposal)
            for proposal, posterior in zip(proposals, self.posteriors)
        ]
        self.proposal_links = ray.get(proposal_processes)

        # if kernel is symmetric, use MTM(II), otherwise use MTM(I).
        if self.kernel.is_symmetric:
            q_x_y = np.zeros(self.k)
        else:
            q_x_y = np.array(
                [
                    self.kernel.get_q(link, proposal_link)
                    for proposal_link in self.proposal_links
                ]
            )

        # get the unnormalised weights.
        self.proposal_weights = np.array(
            [link.posterior + q for link, q in zip(self.proposal_links, q_x_y)]
        )
        self.proposal_weights[np.isnan(self.proposal_weights)] = -np.inf

        # if all posteriors are -Inf, return a random one.
        if np.isinf(self.proposal_weights).all():
            return np.random.choice(self.proposal_links).parameters

        # otherwise, return a random one according to the weights.
        else:
            return np.random.choice(
                self.proposal_links,
                p=np.exp(self.proposal_weights - logsumexp(self.proposal_weights)),
            ).parameters

    def get_acceptance(self, proposal_link, previous_link):
        # check if the proposal makes sense, if not return 0.
        if np.isnan(proposal_link.posterior) or np.isinf(self.proposal_weights).all():
            return 0

        else:
            # create reference proposals.this is fast so no paralellised.
            references = [
                self.kernel.make_proposal(proposal_link) for i in range(self.k - 1)
            ]

            # get the links in parallel.
            reference_processes = [
                posterior.create_link.remote(reference)
                for reference, posterior in zip(references, self.posteriors)
            ]
            self.reference_links = ray.get(reference_processes)

            # if kernel is symmetric, use MTM(II), otherwise use MTM(I).
            if self.kernel.is_symmetric:
                q_y_x = np.zeros(self.k)
            else:
                q_y_x = np.array(
                    [
                        self.kernel.get_q(proposal_link, reference_link)
                        for reference_link in self.reference_links
                    ]
                )

            # get the unnormalised weights.
            self.reference_weights = np.array(
                [link.posterior + q for link, q in zip(self.reference_links, q_y_x)]
            )
            self.reference_weights[np.isnan(self.reference_weights)] = -np.inf

            # get the acceptance probability.
            return np.exp(
                logsumexp(self.proposal_weights) - logsumexp(self.reference_weights)
            )


@ray.remote
class RemotePosterior:
    def __init__(self, posterior):
        self.posterior = posterior

    def create_link(self, parameters):
        return self.posterior.create_link(parameters)

@ray.remote
class ArchiveManager:

    def __init__(self, chain_count):
        # separate collection for each chain
        self.shared_archive = [None] * chain_count
        self.chain_count = chain_count
        self.logger = None
        self.stuck = [False] * chain_count
        self.stuck_counter = STUCK_CHECKING_START
        #self.latest_loglikes = [None] * chain_count
        self.loglikes = [None] * chain_count

    def update_archive(self, samples, chain_id):
        if not isinstance(samples, list) or not isinstance(samples, np.ndarray):
            samples = [samples]
    
        sample_count = len(samples)

        params = np.array([sample.parameters if hasattr(sample, "parameters") else sample for sample in samples])
        params = np.squeeze(params)
        try:
            self.shared_archive[chain_id] = np.vstack((self.shared_archive[chain_id], params))
        except ValueError:
            self.shared_archive[chain_id] = params

        if hasattr(samples[0], "likelihood"):
            likelihoods = np.array([sample.likelihood if hasattr(sample, "likelihood") else sample for sample in samples])
            try:
                self.loglikes[chain_id] = np.vstack((self.loglikes[chain_id], likelihoods))
            except ValueError:
                self.loglikes[chain_id] = likelihoods

        # run stuck check
        self.stuck_counter = self.stuck_counter - sample_count + 1
        self._flag_stuck()

    def get_archive(self):
        try:
            if self.logger is not None:
                delays = ",".join([str(delay) for delay in self.compute_chain_delays()]) + "\n"
                self.log(delays)
            reversed_archive = [a[::-1, :] for a in self.shared_archive]
            return np.concatenate(reversed_archive, axis=0)
            #stacked = np.stack(self.shared_archive)
            #return stacked[:, ::-1, :].reshape(-1, stacked.shape[2])
        except:
            reversed_valid_achive = [a[::-1, :] for a in self.shared_archive if a is not None and len(a) > 0]
            return np.concatenate(reversed_valid_achive, axis=0)
            #stacked = np.stack([a for a in self.shared_archive if a is not None])
            #return stacked[:, ::-1, :].reshape(-1, stacked.shape[2])

    def get_random_subset(self, sample_size):
        """
        Returns a random subset of the archive with the specified sample size.
        If the archive is smaller than the sample size, it returns the whole archive.
        """
        archive = self.get_archive()
        if len(archive) <= sample_size:
            return archive
        else:
            indices = np.random.choice(archive.shape[0], sample_size, replace=False)
            return archive[indices, :]

    def add_logger(self, logger_ref):
        self.logger = logger_ref

    def log(self, message):
        try:
            self.logger.write_to_file.remote(message, "chain_delay")
        except:
            return

    def compute_chain_delays(self):
        try:
            max_length = max([len(a) for a in self.shared_archive])
            delays = [max_length - len(a) for a in self.shared_archive]
        except:
            delays = [0] * self.chain_count
        return delays

    def _get_latest(self):
        """
        Returns the latest sample from each chain.
        If a chain has no samples, it skips that that chain.
        """
        latest_samples = []
        for archive in self.shared_archive:
            print(archive.shape[0] if archive is not None else 0)
            if archive is not None and len(archive) > 0:
                latest_samples.append(archive[-1])
            else:
                latest_samples.append(None)
        return latest_samples

    def _get_generation(self, generation):
        """
        Returns the samples from a specific generation across all chains.
        If a chain does not have that generation, it skips that chain.
        """
        generation_samples = []
        for archive in self.shared_archive:
            if archive is not None and len(archive) > generation:
                generation_samples.append(archive[generation])
            else:
                generation_samples.append(None)
        return generation_samples

    def _get_latest_loglikes(self):
        """
        Returns the latest log-likelihoods from each chain.
        If a chain has no samples, it skips that chain.
        """
        latest_loglikes = []
        for loglike in self.loglikes:
            if loglike is not None and len(loglike) > 0:
                latest_loglikes.append(loglike[-1])
            else:
                latest_loglikes.append([None])
        return np.concatenate(latest_loglikes)

    def _get_generation_loglikes(self, generation):
        """
        Returns the log-likelihoods from a specific generation across all chains.
        If a chain does not have that generation, it skips that chain.
        """
        generation_loglikes = []
        for loglike in self.loglikes:
            if loglike is not None and len(loglike) > generation:
                generation_loglikes.append(loglike[generation])
            else:
                generation_loglikes.append([None])
        return np.concatenate(generation_loglikes)

    def _highest_generation(self):
        """
        Returns the highest generation index across all chains.
        If no chains have samples, it returns -1.
        """
        if not self.shared_archive:
            return -1
        return max([len(archive) - 1 for archive in self.shared_archive if archive is not None and len(archive) > 0], default=-1)

    def _highest_generation_loglike(self):
        """
        Returns the highest generation index across all chains that have log-likelihoods.
        If no chains have log-likelihoods, it returns -1.
        """
        if not self.loglikes:
            return -1
        return max([len(loglike) - 1 for loglike in self.loglikes if loglike is not None and len(loglike) > 0], default=-1)

    def _flag_stuck(self):
        # check if its time to check for stuck chains
        if self.stuck_counter > 0:
            self.stuck_counter = self.stuck_counter - 1
            return

        # reset stuck counter
        self.stuck_counter = self.stuck_counter + STUCK_CHECKING_PERIOD

        # get latest samples
        #latest_samples = self._get_latest()
        # if all chains dont yet have samples, return
        #if len(latest_samples) != self.chain_count:
        #    return
        #logging.info(latest_samples)

        # check what samples are further from posterior compared to the best one
        #loglikes = [link.likelihood for link in latest_samples]
        current_loglikes = self._get_latest_loglikes()
        best_loglike = np.max(current_loglikes)
        bounding_value = best_loglike * STUCK_LIKELIHOOD_THRESHOLD + STUCK_LIKELIHOOD_OFFSET
        behind = [loglike is not None and loglike < bounding_value for loglike in current_loglikes]

        # check what samples have progressed in the last PROGRESSION_CHECK_OFFSET generations
        older_loglikes = self._get_generation_loglikes(self._highest_generation_loglike() - PROGRESSION_CHECK_OFFSET)
        print(older_loglikes, current_loglikes)
        stuck = [older is not None and PROGRESSION_LIKELIHOOD_THRESHOLD * older > current for older, current in zip(older_loglikes, current_loglikes)]

        print(behind, stuck)
        #self.stuck = [b and s for b, s in zip(behind, stuck)]
        self.stuck = np.logical_and(behind, stuck).tolist()
        assert len(self.stuck) == self.chain_count, "Stuck flags do not match chain count"

    def is_stuck(self, chain_id):
        """
        Returns True if the chain is stuck, False otherwise.
        """
        return self.stuck[chain_id]

    def random_nonstuck(self):
        """
        Returns a latest sample from a random non-stuck chain.
        """
        non_stuck_chains = [i for i, s in enumerate(self.stuck) if not s]
        if not non_stuck_chains:
            return None
        random_chain = np.random.choice(non_stuck_chains)
        return self._get_latest()[random_chain]