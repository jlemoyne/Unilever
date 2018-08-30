# Title     : TODO
# Objective : TODO
# Created by: jeanclaudelemoyne
# Created on: 7/31/18
library(ggplot2)

store_demo <- read.csv('/Users/jeanclaudelemoyne/work/Data/Unilever/exogenous/store_demographics.csv')
chosen <- sample(unique(store_demo$store_id), 100)
demo_sample <- subset(store_demo, store_demo$store_id %in% chosen)
print (ls())
store <- c(1:100)
# gg <- ggplot(demo_sample, aes(x=store, y=demo_sample$M2017.35.44)) +
#     geom_point(aes(store,demo_sample$M2017.35.44) +
#     geom_smooth(method="loess", se=F) +
#     xlim(c(0, 1)) +
#     ylim(c(0, 100000)) +
#     labs(subtitle="Male Population 35-44",
#     y="Population",
#     x="Store",
#     title="Male Population 35-44",
#     caption = "Source: store master")

showplot1<-function(indata, inx, iny, header, xlabel) {
    dat <- indata
    p <- ggplot(dat, aes(x=dat[,inx], y=dat[,iny]), environment = environment())
    p <- p + geom_point(size=4, alpha = 0.5) +
        labs(subtitle=header,
        y="Population",
        x=xlabel,
        title="Unilever Retailer Stores Population Density",
        caption = "Source: store master - Office for National Statistcs")
    print(p)
}

showplot2<-function(indata, inx, iny, header, xlabel, ylabel) {
    dat <- indata
    p <- ggplot(dat, aes(x=dat[,inx], y=dat[,iny]), environment = environment())
    p <- p + geom_point(size=4, alpha = 0.5) +
        labs(subtitle=header,
        y=ylabel,
        x=xlabel,
        title="Unilever Retailer Stores Population Density",
        caption = "Source: store master - Office for National Statistcs")
    print(p)
}


showplot1(demo_sample, 1, 10)
