satisfing_sum=[]
# 急需调度时间窗，车辆数量小于5，大于25
a_i = 2
b_i = 5
c_i = 10
d_i = 13
# 需要调度时间窗，车辆数量小于10，大于20
a_j = 2
b_j = 7
c_j = 17
d_j = 22
def TimeWindow(numant,bike_remaining,arriving_time):
    #num = bike_remaining.shape[1]
    num=10
    for j in range(numant):
        for i in range(num-1):
            if bike_remaining[i]<=5 or bike_remaining[i]>=25:#急需调度，最佳时间窗5分钟，最差左右三分钟
                if a_i<=arriving_time[j][i]<b_i:
                    satisfing=(arriving_time[j][i]-a_i)/(b_i-a_i)
                elif b_i<=arriving_time[j][i]<=c_i:
                    satisfing=1
                elif c_i<arriving_time[j][i]<=d_i:
                    satisfing=(arriving_time[j][i]-c_i)/(d_i-c_i)*(-1)+1
                else:
                    satisfing=0
            elif 5<bike_remaining[i]<=10 or 20<=bike_remaining[i]<25:#需要调度，最佳时间窗10分钟，最差左右5分钟
                if a_j<=arriving_time[j][i]<b_j:
                    satisfing=(arriving_time[j][i]-a_j)/(b_j-a_j)
                elif b_j<=arriving_time[j][i]<=c_j:
                    satisfing=1
                elif c_j<arriving_time[j][i]<=d_j:
                    satisfing=(arriving_time[j][i]-c_j)/(d_j-c_j)*(-1)+1
                else:
                    satisfing=0
            else:                                                      #调度时间无要求
                satisfing=1
            satisfing_sum.append(satisfing)
    return satisfing_sum
