particlesJS("particles-js",{

particles:{

number:{
value:70,
density:{
enable:true,
value_area:900
}
},

color:{
value:"#38bdf8"
},

shape:{
type:"circle"
},

opacity:{
value:0.35
},

size:{
value:3,
random:true
},

line_linked:{
enable:true,
distance:160,
color:"#38bdf8",
opacity:0.18,
width:1
},

move:{
enable:true,
speed:1.2,
direction:"none",
random:false,
straight:false,
out_mode:"out",
bounce:false
}

},

interactivity:{

detect_on:"canvas",

events:{

onhover:{
enable:true,
mode:"grab"
},

onclick:{
enable:true,
mode:"push"
},

resize:true

},

modes:{

grab:{
distance:180,
line_linked:{
opacity:.5
}
},

push:{
particles_nb:4
}

}

},

retina_detect:true

});