import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .storage import ObsStorage 

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import MiniBatchKMeans, DBSCAN
    from sklearn.mixture import GaussianMixture
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class SklearnHistoryEstimator:
    def __init__(self, method, latent_dim, random_state=0, kmeans_batch_size=2048, dbscan_eps=0.5, dbscan_min_samples=10):
        self.method = method
        self.latent_dim = latent_dim
        self.random_state = random_state
        self.kmeans_batch_size = kmeans_batch_size
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples

        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for z_method in ['pca', 'kmeans', 'gmm', 'dbscan']")

        self.scaler = StandardScaler()
        self.model = None
        self.fitted = False
        self.dbscan_centroids = None

    def _fit_model(self, features):
        if self.method == 'pca':
            self.model = PCA(n_components=self.latent_dim, random_state=self.random_state)
            self.model.fit(features)
        elif self.method == 'kmeans':
            self.model = MiniBatchKMeans(
                n_clusters=self.latent_dim,
                random_state=self.random_state,
                batch_size=self.kmeans_batch_size,
                n_init=10,
            )
            self.model.fit(features)
        elif self.method == 'gmm':
            self.model = GaussianMixture(
                n_components=self.latent_dim,
                covariance_type='diag',
                random_state=self.random_state,
            )
            self.model.fit(features)
        elif self.method == 'dbscan':
            self.model = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
            labels = self.model.fit_predict(features)
            centroids = []
            unique_labels = [l for l in np.unique(labels) if l != -1]
            for label in unique_labels[:self.latent_dim]:
                mask = labels == label
                centroids.append(features[mask].mean(axis=0))
            self.dbscan_centroids = np.array(centroids, dtype=np.float32) if len(centroids) > 0 else None
        else:
            raise ValueError(f"Unsupported z estimator method: {self.method}")

    def fit(self, flat_history_np):
        scaled = self.scaler.fit_transform(flat_history_np)
        self._fit_model(scaled)
        self.fitted = True

    def transform(self, flat_history_np):
        if not self.fitted:
            return np.zeros((flat_history_np.shape[0], self.latent_dim), dtype=np.float32)
        scaled = self.scaler.transform(flat_history_np)
        if self.method == 'pca':
            z = self.model.transform(scaled)
        elif self.method == 'kmeans':
            z = self.model.transform(scaled)
        elif self.method == 'gmm':
            z = self.model.predict_proba(scaled)
        elif self.method == 'dbscan':
            if self.dbscan_centroids is None or self.dbscan_centroids.shape[0] == 0:
                z = np.zeros((scaled.shape[0], self.latent_dim), dtype=np.float32)
            else:
                dists = np.linalg.norm(scaled[:, None, :] - self.dbscan_centroids[None, :, :], axis=2)
                z = np.zeros((scaled.shape[0], self.latent_dim), dtype=np.float32)
                n_centroids = min(self.latent_dim, dists.shape[1])
                z[:, :n_centroids] = dists[:, :n_centroids]
        else:
            raise ValueError(f"Unsupported z estimator method: {self.method}")

        return z.astype(np.float32)


class ZeroLatentEncoder(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim

    def forward(self, x):
        return torch.zeros(x.shape[0], self.output_dim, device=x.device)


# computes and returns the latent from the expert
class DaggerExpert(nn.Module):
    def __init__(self, loadpth, runid, total_obs_size, T, base_obs_size, nenvs, geomDim = 4, n_futures = 3):
        super(DaggerExpert, self).__init__()
        path = '/'.join([loadpth, 'policy_' + runid + '.pt'])
        self.policy = torch.jit.load(path)
        self.geomDim = geomDim
        self.n_futures = n_futures
        mean_pth = loadpth + "/mean" + runid + ".csv"
        var_pth = loadpth + "/var" + runid + ".csv"
        # Keep a 2D shape even when the CSV has a single row.
        obs_mean = np.atleast_2d(np.loadtxt(mean_pth, dtype=np.float32))
        obs_var = np.atleast_2d(np.loadtxt(var_pth, dtype=np.float32))
        # cut it
        obs_mean = obs_mean[:,obs_mean.shape[1]//2:]
        obs_var = obs_var[:,obs_var.shape[1]//2:]
        self.mean = self.get_tiled_scales(obs_mean, nenvs, total_obs_size, base_obs_size, T)
        self.var = self.get_tiled_scales(obs_var, nenvs, total_obs_size, base_obs_size, T)
        self.tail_size = total_obs_size - (T + 1) * base_obs_size

    def get_tiled_scales(self, invec, nenvs, total_obs_size, base_obs_size, T):
        outvec = np.zeros([nenvs, total_obs_size], dtype = np.float32)
        outvec[:, :base_obs_size * T] = np.tile(invec[0, :base_obs_size], [1, T])
        outvec[:, base_obs_size * T:] = invec[0]
        return outvec

    def forward(self, obs):
        obs = obs[:,-self.tail_size:]
        with torch.no_grad():
            prop_latent = self.policy.prop_encoder(obs[:, :-self.geomDim*(self.n_futures+1)-1]) # since there is also ref at the end
            geom_latents = []
            for i in reversed(range(self.n_futures+1)):
                start = -(i+1)*self.geomDim -1
                end = -i*self.geomDim -1
                if (end == 0):
                    end = None
                geom_latent = self.policy.geom_encoder(obs[:,start:end])
                geom_latents.append(geom_latent)
            geom_latents = torch.hstack(geom_latents)
            expert_latent = torch.cat((prop_latent, geom_latents), dim=1)
        return expert_latent

class DaggerAgent:
    def __init__(self, expert_policy,
                 prop_latent_encoder, student_mlp,
                 T, base_obs_size, device, n_futures=3, train_student_mlp=False):
        expert_policy.to(device)
        if hasattr(prop_latent_encoder, 'to'):
            prop_latent_encoder.to(device)
        #geom_latent_encoder.to(device)
        student_mlp.to(device)
        self.expert_policy = expert_policy
        self.prop_latent_encoder = prop_latent_encoder
        #self.geom_latent_encoder = geom_latent_encoder
        self.student_mlp = student_mlp
        self.base_obs_size = base_obs_size
        self.T = T
        self.device = device
        self.mean = expert_policy.mean
        self.var = expert_policy.var
        self.n_futures = n_futures
        self.itr = 0
        self.current_prob = 0
        # copy expert weights for mlp policy
        self.student_mlp.architecture.load_state_dict(self.expert_policy.policy.action_mlp.state_dict())


        for param in self.expert_policy.policy.parameters():
            param.requires_grad = False
        for param in self.student_mlp.parameters():
            param.requires_grad = train_student_mlp

    def set_itr(self, itr):
        self.itr = itr
        if (itr+1) % 100 == 0:
            self.current_prob += 0.1
            print(f"Probability set to {self.current_prob}")

    def get_history_encoding(self, obs):
        hlen = self.base_obs_size * self.T
        raw_obs = obs[:, : hlen]
        # Hack to add velocity
        #velocity = obs[:, self.velocity_idx] -> Velocity thing is not robust
        #raw_obs[:, -3:] = velocity
        if hasattr(self.prop_latent_encoder, 'transform'):
            raw_obs_np = raw_obs.detach().cpu().numpy()
            prop_latent_np = self.prop_latent_encoder.transform(raw_obs_np)
            prop_latent = torch.from_numpy(prop_latent_np).to(self.device)
        else:
            prop_latent = self.prop_latent_encoder(raw_obs)
        #geom_latent = self.geom_latent_encoder(raw_obs)
        return prop_latent

    def evaluate(self, obs):
        hlen = self.base_obs_size * self.T
        obdim = self.base_obs_size
        prop_latent = self.get_history_encoding(obs)
        #expert_latent = self.get_expert_latent(obs)
        #expert_future_geoms = expert_latent[:,prop_latent.shape[1]+geom_latent.shape[1]:]
        # assume that nothing changed
        #geom_latents = []
        #for i in range(self.n_futures + 1):
        #    geom_latents.append(geom_latent)
        #geom_latents = torch.hstack((geom_latent, expert_future_geoms))
        #if np.random.random() < self.current_prob:
        #    # student action
        output = torch.cat([obs[:, hlen : hlen + obdim], prop_latent], 1)
        #else:
        #    # expert action
        #    output = torch.cat([obs[:, hlen : hlen + obdim], expert_latent], 1)
        output = self.student_mlp.architecture(output)
        return output

    def get_expert_action(self, obs):
        hlen = self.base_obs_size * self.T
        obdim = self.base_obs_size
        expert_latent = self.get_expert_latent(obs)
        output = torch.cat([obs[:, hlen : hlen + obdim], expert_latent], 1)
        #else:
        #    # expert action
        #    output = torch.cat([obs[:, hlen : hlen + obdim], expert_latent], 1)
        output = self.student_mlp.architecture(output)
        return output

    def get_student_action(self, obs):
        return self.evaluate(obs)

    def get_expert_latent(self, obs):
        with torch.no_grad():
            latent = self.expert_policy(obs).detach()
            return latent

    def save_deterministic_graph(self, fname_prop_encoder,
                                 fname_mlp, example_input, device='cpu'):
        hlen = self.base_obs_size * self.T

        if hasattr(self.prop_latent_encoder, 'to'):
            prop_encoder_graph = torch.jit.trace(self.prop_latent_encoder.to(device), example_input[:, :hlen])
        else:
            zero_encoder = ZeroLatentEncoder(self.student_mlp.input_shape[0] - self.base_obs_size).to(device)
            prop_encoder_graph = torch.jit.trace(zero_encoder, example_input[:, :hlen])
        torch.jit.save(prop_encoder_graph, fname_prop_encoder)

        #geom_encoder_graph = torch.jit.trace(self.geom_latent_encoder.to(device), example_input[:, :hlen])
        #torch.jit.save(geom_encoder_graph, fname_geom_encoder)

        mlp_graph = torch.jit.trace(self.student_mlp.architecture.to(device), example_input[:, hlen:])
        torch.jit.save(mlp_graph, fname_mlp)

        if hasattr(self.prop_latent_encoder, 'to'):
            self.prop_latent_encoder.to(self.device)
        #self.geom_latent_encoder.to(self.device)
        self.student_mlp.to(self.device)

class DaggerTrainer:
    def __init__(self,
            actor,
            num_envs, 
            num_transitions_per_env,
            obs_shape, latent_shape,
            base_obs_size,
            history_len,
            num_learning_epochs=4,
            num_mini_batches=4,
            device=None,
            learning_rate=5e-4,
            method='supervised',
            z_method='predictive',
            use_priv_decoder=False,
            priv_decoder_dim=23,
            priv_decoder_weight=1.0,
            policy_loss_weight=0.0,
            detach_z_for_policy=True,
            random_state=0):

        self.actor = actor
        self.method = method
        self.z_method = z_method
        self.base_obs_size = base_obs_size
        self.history_len = history_len
        self.latent_shape = latent_shape
        target_shape = [latent_shape] if self.method == 'supervised' else [base_obs_size]
        self.storage = ObsStorage(num_envs, num_transitions_per_env, [obs_shape], target_shape, device)

        self.optimizer = None
        self.scheduler = None
        if self.method == 'supervised' or self.z_method == 'predictive':
            self.optimizer = optim.Adam([*self.actor.prop_latent_encoder.parameters()], lr=learning_rate)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.1)

        self.use_priv_decoder = use_priv_decoder
        self.priv_decoder_weight = priv_decoder_weight
        self.priv_decoder_dim = priv_decoder_dim
        self.policy_loss_weight = policy_loss_weight
        self.detach_z_for_policy = detach_z_for_policy

        self.priv_decoder = None
        self.priv_decoder_optimizer = None
        if self.use_priv_decoder:
            self.priv_decoder = nn.Sequential(
                nn.Linear(latent_shape, 64), nn.LeakyReLU(),
                nn.Linear(64, priv_decoder_dim)
            ).to(device)
            self.priv_decoder_optimizer = optim.Adam(self.priv_decoder.parameters(), lr=learning_rate)

        self.policy_optimizer = None
        if self.policy_loss_weight > 0.0:
            trainable_mlp = [p for p in self.actor.student_mlp.parameters() if p.requires_grad]
            if len(trainable_mlp) > 0:
                self.policy_optimizer = optim.Adam(trainable_mlp, lr=learning_rate)

        self.sklearn_estimator = None
        if self.method != 'supervised' and self.z_method in ['pca', 'kmeans', 'gmm', 'dbscan']:
            self.sklearn_estimator = SklearnHistoryEstimator(self.z_method, latent_shape, random_state=random_state)
            self.actor.prop_latent_encoder = self.sklearn_estimator
        self.device = device
        self.itr = 0
        self.prev_obs = None

        # env parameters
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.loss_fn = nn.MSELoss()

    def observe(self, obs):
        with torch.no_grad():
            actions = self.actor.get_student_action(torch.from_numpy(obs).to(self.device))
            #actions = self.actor.get_expert_action(torch.from_numpy(obs).to(self.device))
        return actions.detach().cpu().numpy()

    def reset_rollout(self):
        self.prev_obs = None

    def step(self, obs):
        if self.method == 'supervised':
            expert_latent = self.actor.get_expert_latent(torch.from_numpy(obs).to(self.device))
            self.storage.add_obs(obs, expert_latent)
            return

        if self.prev_obs is not None:
            hlen = self.base_obs_size * self.history_len
            next_base_obs = torch.from_numpy(obs[:, hlen:hlen + self.base_obs_size]).to(self.device)
            self.storage.add_obs(self.prev_obs, next_base_obs)
        self.prev_obs = obs.copy()

    def update(self):
        # Learning step
        mse_loss = self._train_step()
        self.storage.clear()
        self.prev_obs = None
        return mse_loss

    def _train_step(self):
        self.itr += 1
        self.actor.set_itr(self.itr)
        if self.storage.step == 0:
            if self.method == 'supervised':
                return {'prop_mse': 0.0, 'geom_mse': 0.0, 'unsup_mse': 0.0, 'decoder_mse': 0.0, 'policy_mse': 0.0}
            return {'prop_mse': 0.0, 'geom_mse': 0.0, 'unsup_mse': 0.0, 'decoder_mse': 0.0, 'policy_mse': 0.0}

        for epoch in range(self.num_learning_epochs):
            # return loss in the last epoch
            prop_mse = 0
            geom_mse = 0
            unsup_mse = 0
            decoder_mse = 0
            policy_mse = 0
            loss_counter = 0

            if self.method != 'supervised' and self.sklearn_estimator is not None:
                hlen = self.base_obs_size * self.history_len
                all_obs = self.storage.obs[:self.storage.step].reshape(-1, self.storage.obs.size(-1))
                history_obs = all_obs[:, :hlen].detach().cpu().numpy()
                self.sklearn_estimator.fit(history_obs)

            for obs_batch, expert_action_batch in self.storage.mini_batch_generator_inorder(self.num_mini_batches):
                if self.method == 'supervised':
                    predicted_prop_latent = self.actor.get_history_encoding(obs_batch)
                    loss_prop = self.loss_fn(predicted_prop_latent[:,:8], expert_action_batch[:,:8])
                    loss_geom = self.loss_fn(predicted_prop_latent[:,8:], expert_action_batch[:,8:])
                    loss = loss_geom + loss_prop
                    unsup_batch_loss = torch.zeros(1, device=self.device)
                else:
                    hlen = self.base_obs_size * self.history_len
                    history_obs = obs_batch[:, :hlen]

                    if self.z_method == 'predictive':
                        predicted_latent, predicted_next_obs = self.actor.prop_latent_encoder.predict_next(history_obs)
                        unsup_batch_loss = self.loss_fn(predicted_next_obs, expert_action_batch)
                        loss = unsup_batch_loss
                    else:
                        predicted_latent = self.actor.get_history_encoding(obs_batch)
                        unsup_batch_loss = torch.zeros(1, device=self.device)
                        loss = torch.zeros(1, device=self.device)

                    loss_prop = torch.zeros(1, device=self.device)
                    loss_geom = torch.zeros(1, device=self.device)

                    if self.use_priv_decoder:
                        priv_start = hlen + self.base_obs_size
                        priv_end = priv_start + self.priv_decoder_dim
                        priv_target = obs_batch[:, priv_start:priv_end]
                        priv_pred = self.priv_decoder(predicted_latent.detach())
                        dec_loss = self.loss_fn(priv_pred, priv_target)
                        if self.priv_decoder_optimizer is not None:
                            self.priv_decoder_optimizer.zero_grad()
                            (self.priv_decoder_weight * dec_loss).backward()
                            self.priv_decoder_optimizer.step()
                        decoder_mse += dec_loss.item()

                    if self.policy_loss_weight > 0.0 and self.policy_optimizer is not None:
                        base_obs = obs_batch[:, hlen:hlen + self.base_obs_size]
                        latent_for_policy = predicted_latent.detach() if self.detach_z_for_policy else predicted_latent
                        student_in = torch.cat([base_obs, latent_for_policy], dim=1)
                        student_action = self.actor.student_mlp.architecture(student_in)
                        with torch.no_grad():
                            expert_action = self.actor.get_expert_action(obs_batch)
                        pol_loss = self.loss_fn(student_action, expert_action)
                        self.policy_optimizer.zero_grad()
                        (self.policy_loss_weight * pol_loss).backward()
                        self.policy_optimizer.step()
                        policy_mse += pol_loss.item()

                # Gradient step
                if self.optimizer is not None:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                prop_mse += loss_prop.item()
                geom_mse += loss_geom.item()
                unsup_mse += unsup_batch_loss.item()
                loss_counter += 1

            avg_prop_loss = prop_mse / loss_counter
            avg_geom_loss = geom_mse / loss_counter
            avg_unsup_loss = unsup_mse / loss_counter
            avg_decoder_loss = decoder_mse / loss_counter
            avg_policy_loss = policy_mse / loss_counter

        if self.scheduler is not None:
            self.scheduler.step()
        return {
            'prop_mse': avg_prop_loss,
            'geom_mse': avg_geom_loss,
            'unsup_mse': avg_unsup_loss,
            'decoder_mse': avg_decoder_loss,
            'policy_mse': avg_policy_loss,
        }
