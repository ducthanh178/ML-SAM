import torch


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    # def load_state_dict(self, state_dict):
    #     super().load_state_dict(state_dict)
    #     self.base_optimizer.param_groups = self.param_groups
    def load_state_dict(self, state_dict):
        # 1. Tải trạng thái chung của SAM (self.defaults, self.param_groups)
        super().load_state_dict(state_dict)
        
        # 2. FIX LỖI LOAD: Tải trạng thái của Base Optimizer (SGD)
        # Dùng hàm load_state_dict của SGD để tải Momentum Buffer từ key mới
        if 'base_optimizer_state_dict' in state_dict:
            self.base_optimizer.load_state_dict(state_dict['base_optimizer_state_dict'])
        else:
            print("Warning: Missing base_optimizer_state_dict. Starting momentum from 0.")
            
        # 3. Đồng bộ lại param_groups 
        # Cần thiết vì SGD có cấu trúc khác Optimizer Wrapper
        self.base_optimizer.param_groups = self.param_groups
        
        # 4. FIX LỖI CŨ: Xử lý reference sau khi tải (để đồng bộ trạng thái)
        # Mặc dù SGD đã load state, nhưng phải đồng bộ trạng thái giữa self.state và self.base_optimizer.state
        self.state = self.base_optimizer.state # Đặt self.state thành reference của base_optimizer.state

    def state_dict(self):
        # Lấy state_dict mặc định của SAM (chứa các param_groups và self.state)
        # Lưu ý: self.state đã chứa state của SGD (Momentum Buffer) do quá trình train
        # đã gọi self.base_optimizer.step(), và SGD đã tạo buffer trong self.state
        sam_state_dict = super().state_dict()
        
        # Thêm trạng thái của base_optimizer vào dictionary
        # Đây là cách chuẩn để đảm bảo Momentum Buffer được lưu
        sam_state_dict['base_optimizer_state_dict'] = self.base_optimizer.state_dict()
        
        return sam_state_dict

