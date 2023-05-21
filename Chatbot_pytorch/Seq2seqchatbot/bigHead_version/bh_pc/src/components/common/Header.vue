<template>
  <div class="header-box">
    <div class="header">
      <div class="content">
        <ul class="nav full-right">
          <li v-for="nav in nav_list">
            <!--站内用这个-->
            <span v-if="nav.is_site"><a :href="nav.link" class="word">{{ nav.title }}</a></span>
            <!--站外用这个 没区别哇？-->
            <span v-else><router-link :to="nav.link" class="word">{{ nav.title }}</router-link></span>
          </li>

          <li>
          <span v-if="token">
              <el-menu class="member" mode="horizontal" >
                <el-submenu index="2">
                  <template slot="title">
                    <router-link to="/user" class="word">个人中心</router-link>
                  </template>
                  <el-menu-item index="2-1">
                    <span @click="logoutHandler" class="second-word">退出登录</span>
                  </el-menu-item>

                  <el-menu-item index="2-2">
                    <span @click="jump" class="second-word">回到首页</span>
                  </el-menu-item>
                </el-submenu>
              </el-menu>
            </span>

            <span v-else>
              <router-link to="/user/login" class="word">登录账号</router-link>
            </span>
          </li>
        </ul>

      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: "Header",
  data() {
    return {
      token: "", // 默认没登录
      nav_list: [],
    }
  },
  options: {
       styleIsolation: 'shared'
   },
  created() {
    this.check_user_login();
    this.get_nav();
  },
  methods: {
    get_nav() {
      this.$axios.get(`${this.$settings.HOST}/index/nav`, {}).then(response => {
        this.nav_list = response.data;
      }).catch(error => {
        console.log(error.response);
      })
    },

    check_user_login() {
      // 获取用户的登录状态
      this.token = sessionStorage.user_token || localStorage.user_token;
      return this.token;
    },
    logoutHandler() {
      // 退出登录，最好别clear()，删除别人的data
      localStorage.removeItem("user_token");
      localStorage.removeItem("user_id");
      localStorage.removeItem("user_name");
      localStorage.removeItem("nickname");
      localStorage.removeItem("avatar");
      sessionStorage.removeItem("user_token");
      sessionStorage.removeItem("user_id");
      sessionStorage.removeItem("user_name");
      sessionStorage.removeItem("nickname");
      sessionStorage.removeItem("avatar");
      this.check_user_login(); // 再次判断登录状态
      this.jump()
    },
     jump() {
      this.$router.push("/")
    }
  },
}
</script>

<style scoped>
.word {
  color: #b5c6c6;
  /*text-shadow: 3px 3px 10px #9cacb6;*/
  background: #47555e;
  font-weight: bold;
  letter-spacing:1px;
}

.second-word {
  /*line-height: 20px;*/
  /*height:20px;*/
  font-size: 18px;
  color: #b5c6c6;
  /*text-shadow: 3px 3px 10px #9cacb6;*/
  background: #47555e;
}

.header-box {
  height: 80px;
}

.header {
  width: 100%;
  height: 80px;
  /*下划线*/
  box-shadow: 0px 0.5px 0.5px 0px #d1f2f7;
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  margin: auto;
  z-index: 99;
  background-color: #47555e;
  /*opacity: 1;*/
}

.header .content {
  max-width: 1200px;
  width: 100%;
  margin: 0 auto;
}

.header .nav li {
  float: left;
  height: 80px;
  line-height: 80px;
  margin-right: 20px;
  font-size: 16px;
  color: #9cacb6;
  cursor: pointer;
}

.header .nav li span {
  padding-bottom: 16px;
  padding-left: 35px;
  padding-right: 5px;
}

.header .nav li span a {
  display: inline-block;
}


.header .nav li :hover{
  color: #d1e6e6;
}

.header .login-bar .login-box span {
  color: #4a4a4a;
  cursor: pointer;
}

.header .login-bar .login-box span:hover {
  color: #000000;
}

.member {
  display: inline-block;
  /*margin-left: 20px;*/
  background: #47555e;
}

.el-menu.el-menu--horizontal {
    /* border-bottom: solid 1px #e6e6e6;  默认导航条下有白边*/
  border:None;
}
/*二级菜单是使用slot添加的，单独给这个slot添加样式，等渲染出来，样式没有了，需要有深选择器。*/
/*https://blog.csdn.net/Laputa219/article/details/115658445*/

/*选中菜单候1、设置鼠标滑过、鼠标点击时，菜单的背景色，2、设置被点击菜单项（菜单展开）的背景色。*/
/*https://blog.csdn.net/sinat_38297809/article/details/118526495*/
.el-submenu /deep/ .el-submenu__title{
    top: 8px;
    padding-left: 0;
    padding-right: 0;
    background: #47555e;
}

.el-submenu.is-opened /deep/  .el-submenu__title {
    background-image: initial;
    background-color: #47555e;
  }

.el-submenu /deep/ .el-submenu__title:focus{
  background-color: #47555e;
}

.el-submenu /deep/ .el-submenu__title:hover{
  background-color: #47555e;
}

/*动态查看css，动态处右键检查*/
.el-menu .el-menu-item, .el-menu--horizontal .el-menu .el-submenu__title {
    background-color:#47555e;
    float: none;
    height: 36px;
    line-height: 36px;
    padding: 0 10px;
    color: #47555e;
   /*几个下拉框之间的间距*/
    margin-top: 0.5px;

}

</style>

