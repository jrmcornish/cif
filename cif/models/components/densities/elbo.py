from .density import Density
from .exact import BijectionDensity


class ELBODensity(Density):
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

    def _elbo(self, x):
        result = self.q_u_density.sample(x)
        u = result["sample"]
        log_q_u = result["log-prob"]

        result = self.bijection.x_to_z(x, u=u)
        z = result["z"]

        log_jac = result["log-jac"]

        log_p_u = self.p_u_density.log_prob(u, z)["log-prob"]

        prior_dict = self.prior.elbo(z)

        return {
            "elbo": log_jac + log_p_u - log_q_u + prior_dict["elbo"],
            "bijection-info": result,
            "prior-dict": prior_dict
        }

    def _fix_random_u(self):
        fixed_prior, z = self.prior._fix_random_u()
        z = z.unsqueeze(0)
        u = self.p_u_density.sample(z)["sample"]
        fixed_bijection = self.bijection.condition(u.squeeze(0))
        new_z = fixed_bijection.z_to_x(z)["x"].squeeze(0)
        return BijectionDensity(prior=fixed_prior, bijection=fixed_bijection), new_z

    def fix_u(self, u):
        fixed_prior = self.prior.fix_u(u=u[1:])
        fixed_bijection = self.bijection.condition(u[0])
        return BijectionDensity(prior=fixed_prior, bijection=fixed_bijection)

    def _sample(self, num_samples):
        z = self.prior.sample(num_samples)
        u = self.p_u_density.sample(z)["sample"]
        return self.bijection.z_to_x(z, u=u)["x"]

    def _fixed_sample(self, noise):
        z = self.prior.fixed_sample(noise=noise)
        u = self.p_u_density.sample(z)["sample"]
        return self.bijection.z_to_x(z, u=u)["x"]
