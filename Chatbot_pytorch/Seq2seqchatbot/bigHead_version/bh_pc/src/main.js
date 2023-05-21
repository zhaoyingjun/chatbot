// Copyright (C) 2023 - 2023 127.0, Inc. All Rights Reserved
// @Time    : 2023/2/12 15:13
// @Author  : 127.0
// @Email   : 2013018259@qq.com
// @File    : bh_pc directory

// The Vue build version to load with the `import` command
// (runtime-only or standalone) has been set in webpack.base.conf with an alias.
import Vue from 'vue'
import App from './App'
import router from './router/index'
import settings from "./settings"
Vue.config.productionTip = false
Vue.prototype.$settings = settings;

import ElementUI from 'element-ui';
import 'element-ui/lib/theme-chalk/index.css';
Vue.use(ElementUI);
import Chat from 'jwchat'
Vue.use(Chat)
import moment from 'moment'
Vue.prototype.$moment = moment;

import axios from 'axios';
axios.defaults.withCredentials = false;
Vue.prototype.$axios = axios;

/* eslint-disable no-new */
new Vue({
  el: '#app',
  router,
  components: { App },
  template: '<App/>'
})
