

#get JOYCE data












    

























joyceDf = getObservationData(measurementDatetime, aolusHlosAngle)


fig = plt.figure(figsize=(20,10))
plt.title("Wind-Speed "+x.strftime("%Y-%m-%d"))
ax = plt.axes()
#plt.scatter(rayleighGdf['speed'],rayleighGdf['alt'], label="Aeolus - Rayleigh")
#plt.scatter(mieGdf['speed'],mieGdf['alt'], label="Aeolus - Mie")
plt.scatter(joyceDf['speed'],joyceDf['alt'], label="JOYCE - Radar/Lidar")
plt.ylim([0,20000])
plt.xlim([-50,50])
ax.set_xlabel("horizontal windspeed [m/s]")
ax.set_ylabel("height AGL [m]")
ax.legend()
plt.savefig(imagePath+x.strftime("%Y-%m-%d")+'_test.png',dpi=150)
plt.show()
