#pragma once
#include <algorithm>
#include <cmath>
#include "kmpc/types.hpp"
namespace kmpc {
// Sampling controls geometry accuracy, not the 20 ms execution clock. Consume
// every complete cell inside the joint-step box, then bisect only the first
// infeasible cell. Never skip an intervening path boundary.
template<class At>
double bounded_path_progress(double start,double end,const Vec18& origin,
                             const Vec18& cap,const At& at) {
  auto fits=[&](double s) {return ((at(s)-origin).cwiseAbs().array()<=cap.array()+1e-12).all();};
  if(!fits(start)) return start;
  double lo=start;
  while(lo<end-1e-12) {
    const double boundary=std::min(end,std::floor(lo+1e-9)+1.0);
    if(fits(boundary)) {lo=boundary;continue;}
    double hi=boundary;
    for(int it=0;it<28;++it) {
      const double mid=(lo+hi)*.5;
      if(fits(mid))lo=mid;else hi=mid;
    }
    break;
  }
  return lo;
}
}
