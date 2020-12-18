library(zoo)
library(parallel)
library(xts)

library(quantmod)
library(rugarch)
library(rmgarch)
library(fBasics)
library(MSwM)
library(forecast)
library(timeSeries)
library(timeSeries)

SPX=read.table("/Users/liuxichen/Downloads/0.csv",header=T,fill = TRUE,sep=',')
 
head(SPX)

SPX_P <- as.numeric(SPX$Price)
S_date <- as.Date(as.character(SPX$Date),"%Y-%m-%d")
SPX <- xts(SPX_P, S_date)

SPX=diff(SPX[,1])/lag(SPX[,1])
SPX<-SPX[-1,]
names(SPX)[1]<-"SPX"





getSymbols('GDP',src='FRED')
GDP=GDP[222:292]

 
GDP=diff(GDP[,1])/lag(GDP[,1])
GDP<-GDP[-1,]
names(GDP)[1]<-"GDP"
basicStats(GDP)

adf.test(GDP,k=20)



plot(GDP,ylab="GDP growth rate",main="GDP growth rate started from 2002")
points(GDP,pch='*')

mod<-lm(GDP ~ 1)
#test candadite models
#lag 0
GDP_intercept <- msmFit(mod, k=2, sw=c(T,T), p=0)
summary(GDP_intercept)
#lag 1
GDP_ar1 <- msmFit(mod, k=2, sw=c(T,T,T), p=1)
summary(GDP_ar1)
#lag 2
GDP_ar2 <- msmFit(mod, k=2, sw=c(T,T,T,T), p=2)
summary(GDP_ar2)
#lag 3
GDP_ar3 <- msmFit(mod, k=2, sw=c(T,T,T,T,T), p=3)
summary(GDP_ar3)
#lag 4
GDP_ar4 <- msmFit(mod, k=2, sw=c(T,T,T,T,T,T), p=4)
summary(GDP_ar4)
GDP_ar5 <- msmFit(mod, k=2, sw=c(T,T,T,T,T,T,T), p=5)
summary(GDP_ar5)
GDP_ar6 <- msmFit(mod, k=2, sw=c(T,T,T,T,T,T,T,T), p=6)
summary(GDP_ar6)






plotProb(GDP_ar3 ,which=1)
plotDiag(GDP_ar3, regime=2, which=2)




# data=========================================================


#Consumer Discretionary

Consumer=read.table("/Users/liuxichen/Downloads/1.csv",header=T,fill = TRUE,sep=',')
 
head(Consumer)
 
Consumer_P <- as.numeric(Consumer$Price)
C_date <- as.Date(as.character(Consumer$Date),"%Y-%m-%d")
Consumer <- xts(Consumer_P, C_date)

Consumer=diff(Consumer[,1])/lag(Consumer[,1])
Consumer<-Consumer[-1,]
names(Consumer)[1]<-"Consumer"
basicStats(Consumer)

#Utilities

Utilities=read.table("/Users/liuxichen/Downloads/2.csv",header=T,fill = TRUE,sep=',')

Utilities_P <- as.numeric(Utilities$Price)
U_date <- as.Date(as.character(Utilities$Date), "%Y-%m-%d")
Utilities <- xts(Utilities_P, U_date)
Utilities=diff(Utilities[,1])/lag(Utilities[,1])
Utilities<-Utilities[-1,]
basicStats(Utilities)



Financial=read.table("/Users/liuxichen/Downloads/3.csv",header=T,fill = TRUE,sep=',')
Financial_P <- as.numeric(Financial$Price)
F_date <- as.Date(as.character(Financial$Date), "%Y-%m-%d")
Financial <- xts(Financial_P, F_date)
Financial=diff(Financial[,1])/lag(Financial[,1])
Financial<-Financial[-1,]
names(Financial)[1]<-"Financial"
basicStats(Financial)




Healthcare=read.table("/Users/liuxichen/Downloads/4.csv",header=T,fill = TRUE,sep=',')

Healthcare_P <- as.numeric(Healthcare$Price)
H_date <- as.Date(as.character(Healthcare$Date), "%Y-%m-%d")
Healthcare <- xts(Healthcare_P, H_date)
Healthcare=diff(Healthcare[,1])/lag(Healthcare[,1])
Healthcare<-Healthcare[-1,]
names(Healthcare)[1]<-"Healthcare"
basicStats(Healthcare)





Info=read.table("/Users/liuxichen/Downloads/5.csv",header=T,fill = TRUE,sep=',')

Info_P <- as.numeric(Info$Price)
I_date <- as.Date(as.character(Info$Date), "%Y-%m-%d")
Info<- xts(Info_P, I_date)
Info=diff(Info[,1])/lag(Info[,1])
Info<-Info[-1,]
names(Info)[1]<-"Info tech"
basicStats(Info)





Real=read.table("/Users/liuxichen/Downloads/6.csv",header=T,fill = TRUE,sep=',')
Real_P <- as.numeric(Real$Price)
R_date <- as.Date(as.character(Real$Date), "%Y-%m-%d")
Real<- xts(Real_P, R_date)
Real=diff(Real[,1])/lag(Real[,1])
Real<-Real[-1,]
names(Real)[1]<-"Real Estate"
basicStats(Real)










data <- data.frame(Consumer, Financial,Utilities,Healthcare,Info,Real)
names(data)[1] <- "Consumer"
names(data)[2] <- "Financial"
names(data)[3] <- "Utilities"
names(data)[4] <- "Healthcare"
names(data)[5] <- "Info Tech"
names(data)[6] <- "Real Estate"


boxplot(data$Consumer,data$Financial,data$Utilities,data$Healthcare,data$Info,data$Real,main="Boxplots for each sector",names=c("Consumer","Financial","Utilities", "Healthcare","Info Tech","Real Estate"),ylab="Simple return",col="orange",border="brown")


par(mfrow=c(2,3))
plot(Consumer,main="CD")
plot(Financial,main="Financials")
plot(Info,main="IT")
plot(Utilities,main="Utilities")
plot(Healthcare,main="Healthcare")
plot(Real,main="Real Estate")

par(mfrow=c(2,1))
acf(Consumer,plot=T)
acf(Consumer^2,plot=T)

Box.test(Consumer^2,lag=10,type='Ljung')


par(mfrow=c(2,1))
acf(Financial,plot=T)
acf(Financial^2,plot=T)

Box.test(Financial^2,lag=100,type='Ljung')



adf.test(Consumer,k=20)
adf.test(Financial,k=20)
adf.test(Info,k=20)
adf.test(Utilities,k=20)
adf.test(Healthcare,k=20)
adf.test(Real,k=20)






auto.arima(Consumer,ic=c('bic'))
#ARIMA(0,0,1)


auto.arima(Financial,ic=c('bic'))
#ARIMA(1,0,0) 

auto.arima(Utilities,ic=c('bic'))
# ARIMA(1,0,0) 

auto.arima(Healthcare,ic=c('bic'))
#ARIMA(0,0,1)

auto.arima(Info,ic=c('bic'))
#ARIMA(2,0,0)

auto.arima(Real,ic=c('bic')) 
#ARIMA(0,0,1) 









C_spec <- ugarchspec(mean.model=list(armaOrder=c(0,1)),distribution.model = "norm")
C_ugfit = ugarchfit(spec = C_spec, data = Consumer)





F_spec <- ugarchspec(mean.model=list(armaOrder=c(1,0)),distribution.model = "norm")
F_ugfit = ugarchfit(spec = F_spec, data = Financial)
 
F_ugfit


U_spec <- ugarchspec(mean.model=list(armaOrder=c(1,0)),distribution.model = "norm")
U_ugfit = ugarchfit(spec = U_spec, data = Utilities) 
U_ugfit



H_spec <- ugarchspec(mean.model=list(armaOrder=c(0,1)),distribution.model = "norm")
H_ugfit = ugarchfit(spec = H_spec, data =Healthcare )
H_ugfit

#Info:
I_spec <- ugarchspec(mean.model=list(armaOrder=c(2,0)),distribution.model = "norm")
I_ugfit = ugarchfit(spec = I_spec, data =Info )
I_ugfit



R_spec <- ugarchspec(mean.model=list(armaOrder=c(0,1)),distribution.model = "norm")
R_ugfit = ugarchfit(spec = R_spec, data =Real )
R_ugfit


#DCC==============================================

uspec.n = multispec(c(C_spec,F_spec,U_spec,H_spec,I_spec,R_spec))
multf = multifit(uspec.n, data)
multf



dcc1_1 = dccspec(uspec = uspec.n, dccOrder = c(1, 1), distribution = 'mvnorm')
 dcc1_1


fit = dccfit(dcc1_1, data = data, fit.control = list(eval.se = TRUE), fit = multf)
fit


rcor(fit, type="R")[,,'2008-12-31']

cov = rcov(fit)

cov[,,dim(cov)[3]]


mean(cov[1,1,])
mean(cov[1,2,])
mean(cov[1,3,])
mean(cov[1,4,])
mean(cov[1,5,])
mean(cov[1,6,])

mean(cov[2,2,])

mean(cov[2,3,])
mean(cov[2,4,])
mean(cov[2,5,])
mean(cov[2,6,])


mean(cov[3,3,])

mean(cov[3,4,])
mean(cov[3,5,])
mean(cov[3,6,])

mean(cov[4,4,])

mean(cov[4,5,])
mean(cov[4,6,])

mean(cov[5,5,])
mean(cov[6,6,])

mean(cov[5,6,])







plot(fit)



plot(rcor(fit, type="R")['Financial','Consumer',], type='l')
plot(as.xts(rcor(fit)['Financial','Consumer',]),main="Financial sector and Consumer sector")

par(mfrow=c(4,1)) 
plot(as.xts(rcor(fit)['Financial','Consumer',]),main="FI and CD")
plot(as.xts(rcor(fit)['Financial','Utilities',]),main="FI and UI")
plot(as.xts(rcor(fit)['Financial','Healthcare',]),main="FI and HC")
plot(as.xts(rcor(fit)['Financial','Info Tech',]),main="FI and IT")

plot(as.xts(rcor(fit)['Financial','Real Estate',]),main="FI and RE")


par(mfrow=c(4,1)) 

plot(as.xts(rcor(fit)['Consumer','Utilities',]),main="CD and UI")
plot(as.xts(rcor(fit)['Consumer','Healthcare',]),main="CD and HC")
plot(as.xts(rcor(fit)['Consumer','Info Tech',]),main="CD and IT")
plot(as.xts(rcor(fit)['Consumer','Real Estate',]),main="CD and RE")

par(mfrow=c(3,1)) 


plot(as.xts(rcor(fit)['Utilities','Healthcare',]),main="UI and HC")
plot(as.xts(rcor(fit)['Utilities','Info Tech',]),main="UI and IT")

plot(as.xts(rcor(fit)['Utilities','Real Estate',]),main="UI and RE")

par(mfrow=c(2,1)) 



plot(as.xts(rcor(fit)['Healthcare','Info Tech',]),main="HC and IT")
plot(as.xts(rcor(fit)['Healthcare','Real Estate',]),main="HC and RE")

plot(as.xts(rcor(fit)['Info Tech','Real Estate',]),main="IT and RE")
