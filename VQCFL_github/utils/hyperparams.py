
# 返回需要的参数 迭代轮次 batch_size 和每隔几轮打印一次结果
def get_hyperparams(dataset,n_SGD):
    if dataset=="MNIST":
        n_iter = 200
        metric_period = 1
        batch_size = 50
    elif dataset[:5] == "CIFAR":
        n_iter = 200
        metric_period = 1
        batch_size = 50
    else:
        n_iter = 200
        metric_period = 1
        batch_size = 50
    return n_iter, batch_size, metric_period


# 返回保存实验的文件名字
def get_file_name(
    run_name:str,
    dataset: str,
    N: int,
    alg: str,
    condition: str,
    seed: int,
    n_SGD: int,
    lr: float,
    decay: float,
    p: float,
    mu: float,
):


    n_iter, batch_size, meas_perf_period = get_hyperparams(dataset, n_SGD)

    file_name = (
        f"{run_name}_{dataset}_{alg}_i{n_iter}_N{n_SGD}_lr{lr}"
        + f"_B{batch_size}_d{decay}_p{p}_m{meas_perf_period}_{seed}"
    )
    if mu != 0.0:
        file_name += f"_{mu}"
    return file_name

def get_filename(
        name: str,
        dataset: str,
        seed: int,
        n_SGD: int ,
        lr: float,
        decay: float,
        p: float,
        alpha:float,
        n_iter:int
):
    n_iter, batch_size, meas_perf_period = get_hyperparams(dataset, n_SGD)
    filename =( f"{name}_{dataset}_SGD_{n_SGD}_lr_{lr}_decay_{decay}"
               + f"_B{batch_size}_d{decay}_p{p}_m{meas_perf_period}_{seed}_alpha{alpha}_iter{n_iter}"
                )
    return filename



def get_CIFAR10_alphas():
    """Return the different alpha considered for the dirichlet distribution"""
    return [0.001, 0.01, 0.1, 1.0, 10.0]