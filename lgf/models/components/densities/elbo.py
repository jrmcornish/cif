from .density import Density


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

    def _sample(self, num_samples):
        z = self.prior.sample(num_samples)
        u = self.p_u_density.sample(z)["sample"]
        return self.bijection.z_to_x(z, u=u)["x"]

    def _fixed_sample(self):
        z = self.prior.fixed_sample()
        u = self.p_u_density.sample(z)["sample"]
        return self.bijection.z_to_x(z, u=u)["x"]
