(function(){var $=document.querySelector.bind(document),$$=document.querySelectorAll.bind(document),menuActive=false
window.onscroll=function(){var scrollPosition=window.pageYOffset||document.documentElement.scrollTop,windowHeight=window.innerHeight||document.documentElement.clientHeight||document.body.clientHeight,navHeight=$('nav').clientHeight
if(scrollPosition>windowHeight-navHeight){$('nav').classList.add('nav-fixed')
$$('nav > .logo, nav > .nav-toggle').forEach(function(el){el.style.visibility='visible'
el.classList.add('show')
el.classList.remove('hide')})}else{$('nav').classList.remove('nav-fixed')
$$('nav > .logo, nav > .nav-toggle').forEach(function(el){el.style.visibility='hidden'
el.classList.add('hide')
el.classList.remove('show')})}}
$('.nav-icon').addEventListener('click',function(){toggleNav()})
$$('.nav-full a').forEach(function(links){links.addEventListener('click',function(){toggleNav()})})
function toggleNav(){$$('.nav-full, main').forEach(function(el){el.classList.toggle('active')})
if(menuActive){$('.nav-icon').querySelector('img:nth-of-type(1)').style.display='inline-block'
$('.nav-icon').querySelector('img:nth-of-type(2)').style.display='none'
menuActive=false}else{$('.nav-icon').querySelector('img:nth-of-type(1)').style.display='none'
$('.nav-icon').querySelector('img:nth-of-type(2)').style.display='inline-block'
menuActive=true}}
$('.logo').addEventListener('click',function(){if($('.nav-full').classList.contains('active')){$$('.nav-full, main').forEach(function(el){el.classList.toggle('active')})}})
$('body').addEventListener('click',function(){if($('.nav-full').classList.contains('active')){$('html').style.overflowY='hidden'}else{$('html').style.overflowY='scroll'}})
function fullMobileViewport(){var element=this,viewportHeight=window.innerHeight,heightChangeTolerance=100
$(window).resize(function(){if(Math.abs(viewportHeight-window.innerHeight)>heightChangeTolerance){viewportHeight=window.innerHeight
update()}})
function update(){element.style.height=(viewportHeight+'px')}
update()}
$$('header').forEach(function(){fullMobileViewport})})()