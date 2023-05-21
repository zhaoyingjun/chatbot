<template>
  <div id="app">
   <router-view/>
  </div>
</template>

<script>
export default {
  name: 'App',
  components:{

  }
}
</script>

<style>
/* 项目的初始化样式 */
body,h1,h2,h3,h4,p,table,tr,td,ul,li,a,form,input,select,option,textarea{
  margin:0;
  padding: 0;
  font-size: 18px;
}
a{
  text-decoration: none;
  color: #9cacb6;
  /*font-weight:bold;*/
  /*text-shadow: 3px 4px 10px #cfdbe3;*/
}
ul,li{
  list-style: none; /* 去掉符号 */
}
table{
  border-collapse: collapse; /* 合并边框 */
}

/* 工具的全局样式 */
.full-right{
  float: right!important;
}

/*Header.vue中隐藏内置在element-ui、但未在html中使用的样式，无法用样式穿透deep，只能在全局样式中使用
 注意要覆盖调内置默认样式，不需要的属性设为0，如padding:0*/
.el-menu--popup-bottom-start {
    /*导航顶部和下拉框之间的距离*/
    margin-top: 0.2px;
  /*和Header发光颜色一样 但这个背景颜色也会应用到下个框和下下框之间，因此必须要设置这里的margin-top
  同时设置.el-submenu__title的margin-top不为0，保证框之间有间隔*/
    background-color: #d1f2f7;

}

.el-menu--popup {
    z-index: 100;
    min-width: 101px;
    margin-left: -10px;
    border: none;
    padding:0;
}

/*箭头下细条*/
.el-menu--horizontal>.el-submenu.is-active .el-submenu__title {
    border-bottom: 2px solid #47555e;
    color: #47555e;}
</style>
