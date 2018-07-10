# AUV-Image-Enhancement
This repository is aimed at enhancing deep under water images.<br>

This repository has 2 parts one is gate image enhancement i.e gate.py and other auv detection that is submarine.py. The results of the two have been saved in result folder.

<h3>What we have done:</h3>
  <ul>
  <li><b>Gate image engancement:</b><p>In this we have used adaptive histogram equalizaton in  YUV color space and contrast stretching independently to differnt channels to make the gates in the images more identifiable and ready for further analysis like detection etc.. The results in gates folder in result forder contains images before and after the operation and homomorphic filtering was not used here because results without it were better.</p>  </li>
  
  <li><b>AUV images:</b><p>In this we have used adaptive histogram equalization in  YUV color space, contrast stretching independently to differnt channels of rgb, homomorphic filtering. Then caany edge detector was used to find the auv and other objects. We also experimented with Log and weiner filters for the detection purpose.</p>  </li>
  
<h3>Places for improvement:</h3>
 <p>In the AUV images we were not very successful with weiner and Log filters. And we also wish to further improve the homomorphic filter and work more with other color spaces.
