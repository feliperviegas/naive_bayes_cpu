BEGIN{sum_mac = 0; sum_mic = 0;}
{
  mac[NR] = $3;
  sum_mac += $3;
}
END{
  mean_mac = sum_mac/NR;
  sd_mac = 0;
  for(i in mac){
    sd_mac += (mac[i]- mean_mac)*(mac[i]- mean_mac);
  }
  sd_mac = sd_mac / (NR - 1);
  sd_mac = sqrt(sd_mac);
  print $1" "$2" " (double) mean_mac" " (double) sd_mac;
}
