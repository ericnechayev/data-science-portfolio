library(gdata)
library(ggplot2)
library(ggfortify)
library(seasonal)
library(urca)
library(egcm)
library(vars)
Russia = read.xls('/Users/Qiu/Desktop/AMS586/Russia_Data.xlsx')
Russia = data.frame(Russia)
colnames(Russia) = c('Time','GDP','GOS','REE','GBC')
ggplot(Russia,aes(Time,GDP,group = 1))+geom_line()
ggplot(Russia,aes(Time,GOS,group = 1))+geom_line()
ggplot(Russia,aes(Time,REE,group = 1))+geom_line()
ggplot(Russia,aes(Time,GBC,group = 1))+geom_line()

#Seasonal adjustment by using X-13ARIMA-Seats 
GDP_ts = ts(Russia$GDP,start=c(2003,1),frequency = 4)
GOS_ts = ts(Russia$GOS,start=c(2003,1),frequency = 4)
GDP_AD = seas(GDP_ts)
GOS_AD = seas(GOS_ts)
plot(GDP_AD)
plot(GOS_AD)
Adjust_GDP = data.frame(as.numeric(GDP_AD$data[,1]))
Adjust_GOS = data.frame(as.numeric(GOS_AD$data[,1]))

#Due to different scale, we use log to handle data
Russia2 = data.frame(Time = Russia$Time, Ad_GDP = log(Adjust_GDP),Ad_GOS = log(Adjust_GOS),
                     REE = log(Russia$REE), GBC = log(Russia$GBC))
colnames(Russia2) = c('Time','Ad_GDP','Ad_GOS','REE','GBC')
ggplot(Russia2,aes(Time, Ad_GDP, group = 1))+geom_line()
ggplot(Russia2,aes(Time, Ad_GOS, group = 1))+geom_line()

p = ceiling(12*(length(Russia$Time)/100)^0.25)
#Test I(1)
adf1 = summary(ur.df(Russia2$Ad_GDP,type = c('trend'),lags = p, selectlags = c('BIC')))
print(adf1)
adf2 = summary(ur.df(Russia2$Ad_GOS,type = c('trend'),lags = p, selectlags = c('BIC')))
print(adf2)
adf4 = summary(ur.df(Russia2$REE,type = c('trend'),lags = p, selectlags = c('BIC')))
print(adf4)
adf5 = summary(ur.df(Russia2$GBC,type = c('trend'),lags = p, selectlags = c('BIC')))
print(adf5)

#Test I(0)
ddf1 = summary(ur.df(diff(Russia2$Ad_GDP),type = c('trend'),lags = p, selectlags = c('BIC')))
print(ddf1)
ddf2 = summary(ur.df(diff(Russia2$Ad_GOS),type = c('trend'),lags = p, selectlags = c('BIC')))
print(ddf2)
ddf4 = summary(ur.df(diff(Russia2$REE),type = c('trend'),lags = p, selectlags = c('BIC')))
print(ddf4)
ddf5 = summary(ur.df(diff(Russia2$GBC),type = c('trend'),lags = p, selectlags = c('BIC')))
print(ddf5)

#Johansen test: Trace test
VARselect(Russia2[1:52,c(-1)],lag.max = 5)
Jotest=ca.jo(Russia2[1:52,c(-1)], type="trace", K=2, ecdet="const", spec="transitory")
print(summary(Jotest))
#We found at least 1 cointegration relationship

#Search for cointegration
Ans = matrix(rep(NA,16),nrow = 4)
for( i in 1:4){
    for(j in 1:4){
        if(i!=j){
            T = egcm(Russia2[1:52,i+1],Russia2[1:52,j+1])
            Ans[i,j]= as.numeric(is.cointegrated(T))
        }else{
            Ans[i,j] = 0
        }
    }
}
print(Ans)
#We find GBC and REE are conintegrated

#Check the direction of conintegration relationship
print(VARselect(Russia2[1:48,c(4,5)], lag.max = 5, type = "both"))
var <- VAR(Russia2[1:48,c(4,5)], p = 2, type = "both")
causality(var, cause = "REE")$Granger
causality(var, cause = "GBC")$Granger
#REE granger cause GBC. At the same time, GBC also granger cause REE

#Fit Vector Error Correction Model
fit=ca.jo(Russia2[1:48,c(4,5)], type="trace", K=2, ecdet="const", spec="longrun")
vecm = vec2var(fit)
print(vecm)
#Both residual are I(0)
r1 = summary(ur.df(vecm$resid[,1]))
r2 = summary(ur.df(vecm$resid[,2]))
print(r1)
print(r2)

Vecm_Pred = predict(vecm)
Vecm_P_REE = Vecm_Pred$fcst$REE[1:7,1:3]
Vecm_P_GBC = Vecm_Pred$fcst$GBC[1:7,1:3]

Vecm_REE_P = rbind(data.frame(Time = Russia$Time[49:55], Value = Vecm_P_REE[,1], 
                              Len = 'Forecast'),
                   data.frame(Time = Russia$Time[49:55], Value = Vecm_P_REE[,2], Len = 'Lower'),
                   data.frame(Time = Russia$Time[49:55], Value = Vecm_P_REE[,3], Len = 'Upper'),
                   data.frame(Time = Russia$Time, Value = Russia2$REE, Len = 'Real'))
ggplot(Vecm_REE_P, aes(x = Time, y = Value, group = Len, color = Len)) + geom_line()

Vecm_GBC_P = rbind(data.frame(Time = Russia$Time[49:55], Value = Vecm_P_GBC[,1], 
                              Len = 'Forecast'),
                   data.frame(Time = Russia$Time[49:55], Value = Vecm_P_GBC[,2], Len = 'Lower'),
                   data.frame(Time = Russia$Time[49:55], Value = Vecm_P_GBC[,3], Len = 'Upper'),
                   data.frame(Time = Russia$Time, Value = Russia2$GBC, Len = 'Real'))
ggplot(Vecm_GBC_P, aes(x = Time, y = Value, group = Len, color = Len)) + geom_line()

#Vector Autoregressive. Difference all variable into I(0)
Russia3 = data.frame(Time = Russia2$Time[-1],Diff_GDP = na.omit(diff(Russia2$Ad_GDP)),
                     Diff_GOS = na.omit(diff(Russia2$Ad_GOS)),
                     REE = na.omit(diff(Russia2$REE)), 
                     GBC = na.omit(diff(Russia2$GBC)))

ggplot(Russia3,aes(Time,Diff_GDP,group = 1))+geom_line()
ggplot(Russia3,aes(Time,Diff_GOS,group = 1))+geom_line()
ggplot(Russia3,aes(Time,REE,group = 1))+geom_line()
ggplot(Russia3,aes(Time,GBC,group = 1))+geom_line()

#Select lags
print(VARselect(Russia3[1:48,-1], type = 'const',lag.max = 5))
Var_Mdl = VAR(Russia3[1:48,-1], lag.max = 1, type = 'const')

#Test stabiltiy, and all eigenvalue less than one which mean our model stationary
roots(Var_Mdl)

#ACF of residuals
acf(residuals(Var_Mdl))

#Mutivariate Jarque-Bera test
print(normality.test(Var_Mdl, multivariate.only = TRUE))

#Acf
#The null hypothesis is that there is acf of any order up to p
serial.test(Var_Mdl, lags.pt = 16, type = "PT.adjusted")
plot(Var_Mdl)

Var_Pred = predict(Var_Mdl)
P_GDP = Var_Pred$fcst$Diff_GDP[1:6,1:3]
P_GOS = Var_Pred$fcst$Diff_GOS[1:6,1:3]
P_REE = Var_Pred$fcst$REE[1:6,1:3]
P_GBC = Var_Pred$fcst$GBC[1:6,1:3]

Var_GDP_P = rbind(data.frame(Time = Russia3$Time[49:54], Value = P_GDP[,1], 
                             Len = 'Forecast'),
                  data.frame(Time = Russia3$Time[49:54], Value = P_GDP[,2], Len = 'Lower'),
                  data.frame(Time = Russia3$Time[49:54], Value = P_GDP[,3], Len = 'Upper'),
                  data.frame(Time = Russia3$Time, Value = Russia3$Diff_GDP, Len = 'Real'))
ggplot(Var_GDP_P, aes(x = Time, y = Value, group = Len, color = Len)) + geom_line()

Var_GOS_P = rbind(data.frame(Time = Russia3$Time[49:54], Value = P_GOS[,1], 
                             Len = 'Forecast'),
                  data.frame(Time = Russia3$Time[49:54], Value = P_GOS[,2], Len = 'Lower'),
                  data.frame(Time = Russia3$Time[49:54], Value = P_GOS[,3], Len = 'Upper'),
                  data.frame(Time = Russia3$Time, Value = Russia3$Diff_GOS, Len = 'Real'))
ggplot(Var_GOS_P, aes(x = Time, y = Value, group = Len, color = Len)) + geom_line()

Var_REE_P = rbind(data.frame(Time = Russia3$Time[49:54], Value = P_GDP[,1], 
                             Len = 'Forecast'),
                  data.frame(Time = Russia3$Time[49:54], Value = P_REE[,2], Len = 'Lower'),
                  data.frame(Time = Russia3$Time[49:54], Value = P_REE[,3], Len = 'Upper'),
                  data.frame(Time = Russia3$Time, Value = Russia3$REE, Len = 'Real'))
ggplot(Var_REE_P, aes(x = Time, y = Value, group = Len, color = Len)) + geom_line()

Var_GBC_P = rbind(data.frame(Time = Russia3$Time[49:54], Value = P_GBC[,1], 
                             Len = 'Forecast'),
                  data.frame(Time = Russia3$Time[49:54], Value = P_GBC[,2], Len = 'Lower'),
                  data.frame(Time = Russia3$Time[49:54], Value = P_GBC[,3], Len = 'Upper'),
                  data.frame(Time = Russia3$Time, Value = Russia3$GBC, Len = 'Real'))
ggplot(Var_GBC_P, aes(x = Time, y = Value, group = Len, color = Len)) + geom_line()


