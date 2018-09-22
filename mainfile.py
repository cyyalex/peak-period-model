import numpy as np
from aco import measure
import random
import matplotlib.pyplot as plt
from timewindow import Time


population_size=10 #种群数量
generations=5#遗传迭代次数
numant = 20  # 蚂蚁个数
iter = 0  # 迭代初始
itermax =10  # 迭代总数
citynum=10  #站桩数量
pc=0.88 #交配概率
pm=0.3 #变异概率
sw=0.7  #用户满意度权重
lw=1-sw  #路径权重


fitness =[] #适应度
population =[] #种群对应的十进制数值
fitness_sum=[]
optimum_solution=[] #每次迭代所获得的最优解
population_proportion=[]#每个染色体适应度总和的比
shortest = []
shortestlength=[]
all_list = []
BestPopulation=[]
# shape[0]=52 城市个数,也就是任务个数
Q = 1  # 完成率

cum=[]  #累加时间
user_satisfing=[]   #用户满意度


#早晨高峰时期为8.00-8.30

# 站点可租赁个数
bike_available = np.random.randint(20, 30, (1, 10))

# 租赁者个数
renters = np.random.randint(0, 20, (1, 10))

# 租赁后剩余个数
#bike_remaining = bike_available - renters
bike_remaining=[4,12,10,24,26,14,19,25,9,16]
#生成初始种群 α∈（0,5） β∈（2,5） ρ∈（0,1）
def init_population():
    N = np.random.uniform( size=(population_size, 4))   #做一个5X4的随机矩阵
    for i in range(population_size):
        N[i][0] = i + 1   #序号列
        N[i][1] = np.random.uniform(0,2)       #α∈（0,2）
        N[i][2] = np.random.uniform(2, 5)      #β∈（2,5）
        N[i][3] = np.random.uniform(0, 1)        #ρ∈（0,1）
    population.extend(N)
    return population  # 信息素重要程度因子# 启发函数重要程度因子# 信息素的挥发速度


#位置信息
def aco():
    global alpha, beta, rho, lengthbest, lengthaver, pathbest, numcity, coordinates,useraver,pathtable
    coordinates = np.array([[565.0, 575.0], [25.0, 185.0], [345.0, 750.0],[945.0, 685.0], [845.0, 655.0],[760.0, 350],
                            [20.0, 525.0], [525.0, 1000.0], [700.0, 850.0],[235.0, 380.0]])
    path = []  # 保留每只蚂蚁走过各个站桩的时间
    numcity = coordinates.shape[0]
    distmat=measure.getdistmat(numcity,coordinates)
    timemat=measure.GetTimeMat(numcity,coordinates)   #各城市间运输所需时间

    etatable = 1.0 / (timemat + np.diag([1e10] * numcity))
    # diag(),将一维数组转化为方阵 启发函数矩阵，表示蚂蚁从城市i转移到城市j的期望程度
    pheromonetable = np.ones((numcity, numcity))
    # 信息素矩阵 52*52
    pathtable = np.zeros((numant, numcity)).astype(int)
    pathtable_list=np.zeros((numant*itermax*population_size,numcity)).astype(int)

    length_list=[]
    # 路径记录表，转化成整型 40*52

    useraver=np.zeros(itermax)
    lengthaver = np.zeros(itermax)  # 迭代50次，存放每次迭代后，路径的平均长度  50*1
    lengthbest = np.zeros(itermax)  # 迭代50次，存放每次迭代后，最佳路径长度  50*1
    pathbest = np.zeros((itermax,numcity))  # 迭代50次，存放每次迭代后，最佳路径城市的坐标 50*52
    for q in range(population_size):  # 五组参数在蚁群算法中循环
        alpha = population[q][1]
        beta = population[q][2]
        rho = population[q][3]

        for iter in range(itermax): # 迭代总数
            # n个蚂蚁随机放置于m个城市中
            if numant <= numcity:  # 城市数比蚂蚁数多，不用管
                pathtable[:, 0] = np.random.permutation(range(numcity))[:numant]
                # 返回一个打乱的40*52矩阵，但是并不改变原来的数组,把这个数组的第一列(40个元素)放到路径表的第一列中
                # 矩阵的意思是哪个蚂蚁在哪个城市,矩阵元素不大于52
            else:  # 蚂蚁数比城市数多，需要有城市放多个蚂蚁
                pathtable[:numcity, 0] = np.random.permutation(range(numcity))[:]
                # 先放52个
                pathtable[numcity:, 0] = np.random.permutation(range(numcity))[:numant - numcity]
                # 再把剩下的放完
            # print(pathtable[:,0])
            length = np.zeros(numant)  # 1*40的数组

            # 本段程序算出每只/第i只蚂蚁转移到下一个城市的概率
            for i in range(numant):

                # i=0
                visiting = pathtable[i, 0]  # 当前所在的城市
                # set()创建一个无序不重复元素集合
                #visited = set() #已访问过的城市，防止重复
                #visited.add(visiting) #增加元素
                #print(visited)
                unvisited = set(range(numcity))
                # 未访问的城市集合
                # 剔除重复的元素
                unvisited.remove(visiting)  # 删除已经访问过的城市元素

                for j in range(1, numcity):  # 循环numcity-1次，访问剩余的所有numcity-1个城市
                    # j=1
                    # 每次用轮盘法选择下一个要访问的城市
                    listunvisited = list(unvisited)
                    # 未访问城市数,list
                    probtrans = np.zeros(len(listunvisited))
                    # 每次循环都初始化转移概率矩阵1*52,1*51,1*50,1*49....

                    # 以下是计算转移概率
                    for k in range(len(listunvisited)):
                        probtrans[k] = (pheromonetable[visiting][listunvisited[k]]**alpha) \
                                       * (etatable[visiting][listunvisited[k]]**alpha)
                    # eta-从城市i到城市j的启发因子 这是概率公式的分母   其中[visiting][listunvis[k]]是从本城市到k城市的信息素
                    cumsumprobtrans = (probtrans / sum(probtrans)).cumsum()
                    # 求出本只蚂蚁的转移到各个城市的概率斐波衲挈数列

                    cumsumprobtrans -= np.random.rand()
                    # 随机生成下个城市的转移概率，再用区间比较
                    # k = listunvisited[ndarray.find(cumsumprobtrans > 0)[0]]
                    k = listunvisited[list(cumsumprobtrans > 0).index(True)]
                    # k = listunvisited[np.where(cumsumprobtrans > 0)[0]]
                    # where 函数选出符合cumsumprobtans>0的数
                    # 下一个要访问的城市

                    pathtable[i, j] = k
                    #print(pathtable)
                    # 采用禁忌表来记录蚂蚁i当前走过的第j城市的坐标，这里走了第j个城市.k是中间值
                    unvisited.remove(k)
                    # visited.add(k)
                    # 将未访问城市列表中的K城市删去，增加到已访问城市列表中

                    length[i] += distmat[visiting][k]
                    path.append(timemat[visiting][k])    #记录各个站桩之间运输时间

                    # 计算本城市到K城市的距离
                    visiting = k


                length[i] += distmat[visiting][pathtable[i, 0]]

            for a in range(numant):                    #将每轮每只蚂蚁路径表放入总表中   构成种群路径表
                for c in range(numcity):
                    pathtable_list[(q*(itermax*numant))+(a+iter*numant)][c]=pathtable[a][c]

                # 计算本只蚂蚁的总的路径距离，包括最后一个城市和第一个城市的距离
            #print('pathtalbe',pathtable)
            #print("ants all length:",length)
            # 包含所有蚂蚁的一个迭代结束后，统计本次迭代的若干统计参数（每只蚂蚁遍历完城市后总时间）

            #本部分求用户满意度
            a = np.zeros((numant, citynum - 1))  # 获取每只蚂蚁到各个站点时间
            for z in range(numant):
                for x in range(citynum - 1):
                    a[z][x] = path[10 * z + x - z]      #a为每只蚂蚁路径表
            arriving_time = np.cumsum(a, axis=1) #每只蚂蚁到达各个站点的时间
            all_satisfing=Time.TimeWindow(numant, bike_remaining, arriving_time)  # 每个站点的用户满意度 五个蚂蚁  每个蚂蚁九个站点

            #print('all_satisfing',all_satisfing)
            b = np.zeros((numant, citynum - 1))  # 每只蚂蚁的总用户满意度
            for z in range(numant):
                for x in range(citynum - 1):
                    b[z][x] = all_satisfing[10 * z + x - z]
            user_satisfing.extend(b.sum(axis=1))

            length_list.extend(length)             #种群长度表


            #lengthaver[iter] = length.mean()

            #print("pathtable", pathtable)
            # 本轮的平均路径

            all_satisfing.clear()          #清除每次迭代后的时间窗列表
            path.clear()  # 清除每次迭代后的路径列表
            # 本部分是为了求出最优用户满意度，返回索引值

            # 如果是第一轮路径，则选择本轮最短的路径,并返回索引值下标，并将其记录
            #print('pathtable',pathtable)
            # 此部分是为了更新信息素
            changepheromonetable = np.zeros((numcity, numcity))
            for i in range(numant):  # 更新所有的蚂蚁
                for j in range(numcity - 1):
                    changepheromonetable[pathtable[i, j]][pathtable[i, j + 1]] += Q / distmat[pathtable[i, j]][
                        pathtable[i, j + 1]]
                    # 根据公式更新本只蚂蚁改变的城市间的信息素      Q/d   其中d是从第j个城市到第j+1个城市的距离
                changepheromonetable[pathtable[i, j + 1]][pathtable[i, 0]] += Q / distmat[pathtable[i, j + 1]][
                    pathtable[i, 0]]
                # 首城市到最后一个城市 所有蚂蚁改变的信息素总和

            # 信息素更新公式p=(1-挥发速率)*现有信息素+改变的信息素
            pheromonetable = (1 - rho) * pheromonetable + changepheromonetable

            iter += 1  # 迭代次数指示器+1
            #print("this iteration end：", iter)
            # 观察程序执行进度，该功能是非必须的


        #比较用户满意度和路径长度
        user_max = user_satisfing[0]
        user_length = length_list[0]
        user_pathtable = pathtable_list[0]
        for i in range(numant*itermax):              #选出用户满意度最高
            if user_satisfing[i]>user_max:
                user_max=user_satisfing[i]
                user_length=length_list[i]
                user_pathtable=pathtable_list[q*numant*itermax+i]
        for j in range(numant*itermax):             #选出用户满意度一样，路径更短
            if user_satisfing[j]==user_max and length_list[j]<user_length:
                user_max = user_satisfing[j]
                user_length = length_list[j]
                user_pathtable = pathtable_list[q*numant*itermax+j]



        length_list.clear()
        user_satisfing.clear()




        # 制作列表【【用户满意度，alpha，beta，rho，路径长度，路径表】，【。。。】。。。】解决对应不一致问题
        each_list = [alpha,beta,rho,user_max,user_length,user_pathtable]
        all_list.append(each_list)
    print(all_list)



#适应值评价
def calculate_fitness():
    for i in range(population_size):
        function_value=(sw*all_list[i][3]*0.01)+(lw*(1/all_list[i][4]))              #0.7权重的用户满意度  0.3的路径长度
        fitness.append(function_value)






#获取每轮最大用户满意度和最短路径种群
def best_value():
    population_bestlist=all_list[0]
    max_satisfitness=all_list[0][3]
    min_lengfitness=all_list[0][4]
    for i in range(population_size):
        if all_list[i][3]>max_satisfitness:
            max_satisfitness=all_list[i][3]
            min_lengfitness=all_list[i][4]
            population_bestlist=all_list[i]
    for j in range(population_size):
        if all_list[j][3]==max_satisfitness and all_list[j][4]<min_lengfitness:
            population_bestlist=all_list[j]
    optimum_solution.append(population_bestlist)
    print('每轮最优种群',optimum_solution)

#选取总迭代后最优解
def ChooseBest():
    best_population=optimum_solution[0]
    best_satising=optimum_solution[0][3]
    best_length=optimum_solution[0][4]
    for i in range(generations):
        if optimum_solution[i][3]>best_satising:
            best_satising=optimum_solution[i][3]
            best_length=optimum_solution[i][4]
            best_population=optimum_solution[i]
    for j in range(generations):
        if optimum_solution[j][3]==best_satising and optimum_solution[j][4]<best_length:
            best_population=optimum_solution[j]
    BestPopulation.extend(best_population)
    print('最优种群的各个参数和解',best_population )
    return best_population[0],best_population[1],best_population[2],best_satising,best_population[4]


#采用轮盘赌算法进行选择过程
def selection():
    fitness_sum = 0
    for i in range(population_size):
        fitness_sum += fitness[i]
        # 计算生存率
    for i in range(population_size):
        population_proportion.append(fitness[i] / fitness_sum)
    pie_fitness = []
    cumsum = 0.0
    for i in range(population_size):
        pie_fitness.append(cumsum + population_proportion[i])
        cumsum += population_proportion[i]
    # 生成随机数在轮盘上选点[0, 1)
    random_selection = []
    for i in range(population_size):
        random_selection.append(random.random())
    # 选择新种群
    new_population = []
    random_selection_id = 0
    global population
    for i in range(population_size):
        while random_selection_id < population_size and random_selection[random_selection_id] < pie_fitness[i]:
            new_population.append(population[i])
            random_selection_id += 1
    population = new_population
    #print(population)          #输出新种群



#进行交配
def crossover():
    for i in range(0,population_size-1,2):
        if random.random()<pc:
            #随机选择交叉点
            change_point=random.randint(1,3)
            temp1=[]
            temp2=[]
            temp1.extend(population[i][0:change_point])
            temp1.extend(population[i+1][change_point:])
            temp2.extend(population[i+1][0:change_point])
            temp2.extend(population[i][change_point:])
            population[i]=temp1
            population[i+1]=temp2
    #print(population[i])
    #print(population[i+1])


def mutation():
    for i in range(population_size):
        if random.random()<pm:
            mutation_point=random.randint(1,3)#随机变异点
            if mutation_point==1:      #如果α变异
                population[i][mutation_point]=random.uniform(0,2)
            else:
                if mutation_point==2:       #如果β变异
                    population[i][mutation_point] = random.uniform(2,5)
                else:                   #如果ρ变异
                    population[i][mutation_point]=random.uniform(0,1)

# 以下是做图部分
# 做出平均路径长度和最优路径长度
def pictrue():
    # fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))
    # axes[0].plot(lengthbest, 'k', marker='*')
    # axes[0].set_title('Best length')
    #axes[0].set_xlabel(u'iteration')

    # 线条颜色black https://blog.csdn.net/ywjun0919/article/details/8692018
    # axes[1].plot(best_value(), 'k', marker='<')
    # axes[1].set_title('Best Satisfing')
    # axes[1].set_xlabel(u'iteration')
    # fig.savefig('Average_Best.png', dpi=500, bbox_inches='tight')
    # plt.close()
    #fig.show()

    # 作出找到的最优路径图
    bestpath = BestPopulation[5]

    plt.plot(coordinates[:, 0], coordinates[:, 1], 'r.', marker='>')
    plt.xlim([-100, 1300])
    # x范围
    plt.ylim([-100, 1300])
    # y范围

    for i in range(numcity - 1):
        # 按坐标绘出最佳两两城市间路径
        m, n = int(bestpath[i]), int(bestpath[i + 1])
        print("best_path:", m, n)
        plt.plot([coordinates[m][0], coordinates[n][0]], [coordinates[m][1], coordinates[n][1]], 'k')

    plt.plot([coordinates[int(bestpath[0])][0], coordinates[int(bestpath[9])][0]],
             [coordinates[int(bestpath[0])][1], coordinates[int(bestpath[8])][1]], 'b')

    ax = plt.gca()
    ax.set_title("Best Path")
    ax.set_xlabel('X_axis')
    ax.set_ylabel('Y_axis')

    plt.savefig('Best Path.png', dpi=500, bbox_inches='tight')
    plt.show()

init_population()
for step in range(generations):
    aco()
    calculate_fitness()
    best_value()

    selection()
    fitness.clear()
    crossover()
    mutation()
    all_list.clear()
ChooseBest()
alpha,beta,rho,max_usersatisfing,min_length=ChooseBest()

pictrue()
print("最高用户满意度",max_usersatisfing)
print('最短路径',min_length)
print('alpha:',alpha)
print('beta:',beta)
print('rho:',rho)
