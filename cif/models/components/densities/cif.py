from .density import Density
from .flow import FlowDensity


# TODO: We could subsume this and the ExactDensities as special cases of
# MarginalDensity by thinking of the prior for MarginalDensity as jointly
# over (z, u) here
class CIFDensity(Density):
    def __init__(
            self,
            prior,
            p_u_density,
            bijection,
            q_u_density
    ):
        super().__init__()
        self.bijection = bijection
        self.prior = prior
        self.p_u_density = p_u_density
        self.q_u_density = q_u_density

    def p_parameters(self):
        return [
            *self.bijection.parameters(),
            *self.p_u_density.parameters(),
            *self.prior.p_parameters()
        ]

    def q_parameters(self):
        result = list(self.q_u_density.parameters())

        prior_q_params = list(self.prior.q_parameters())
        result += prior_q_params

        # If the prior doesn't have any q parameters, then the bijection here
        # also isn't a part of q(u|x) (see equation (15) in the paper)
        if prior_q_params:
            result += list(self.bijection.parameters())

        return result

    def _elbo(self, x, detach_q_params, detach_q_samples):
        result = self.q_u_density.sample(
            cond_inputs=x,
            detach_params=detach_q_params,
            detach_samples=detach_q_samples
        )
        u = result["sample"]
        log_q_u = result["log-prob"]

        result = self.bijection.x_to_z(x, u=u)
        z = result["z"]

        log_jac = result["log-jac"]

        result = self.p_u_density.log_prob(inputs=u, cond_inputs=z)
        log_p_u = result["log-prob"]

        prior_dict = self.prior(
            "elbo",
            z,
            detach_q_params=detach_q_params,
            detach_q_samples=detach_q_samples
        )

        return {
            "log-p": log_jac + log_p_u + prior_dict["log-p"],
            "log-q": log_q_u + prior_dict["log-q"],
            "bijection-info": result,
            "prior-dict": prior_dict
        }

    def _fix_random_u(self):
        fixed_prior, z = self.prior._fix_random_u()
        z = z.unsqueeze(0)
        u = self.p_u_density.sample(z)["sample"]
        fixed_bijection = self.bijection.condition(u.squeeze(0))
        new_z = fixed_bijection.z_to_x(z)["x"].squeeze(0)
        return FlowDensity(prior=fixed_prior, bijection=fixed_bijection), new_z

    def fix_u(self, u):
        fixed_prior = self.prior.fix_u(u=u[1:])
        fixed_bijection = self.bijection.condition(u[0])
        return FlowDensity(prior=fixed_prior, bijection=fixed_bijection)

    def _sample(self, num_samples):
        z = self.prior.sample(num_samples)
        u = self.p_u_density.sample(z)["sample"]
        return self.bijection.z_to_x(z, u=u)["x"]

    def _fixed_sample(self, noise):
        z = self.prior.fixed_sample(noise=noise)
        u = self.p_u_density.sample(z)["sample"]
        return self.bijection.z_to_x(z, u=u)["x"]
