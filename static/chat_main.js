function formatAMPM(date) {
  var hours = date.getHours();
  var minutes = date.getMinutes();
  var ampm = hours >= 12 ? 'PM' : 'AM';
  hours = hours % 12;
  hours = hours ? hours : 12; // the hour '0' should be '12'
  minutes = minutes < 10 ? '0'+minutes : minutes;
  var strTime = hours + ':' + minutes + ' ' + ampm;
  return strTime;
}

var prev_q = "";
var prev_a = "";
var bot_id = "";
var bot_im_url = "";

$(document).ready(function() {
  bot_id = $('#bot_id').text();
  bot_im_url = $('#bot_im_url').text();
});

$(document).keypress(function(e) {
    if(e.which == 13) {
        var ms = {
        username:'arijit',
        name: 'Arijit',
        avatar: 'https://bootdey.com/img/Content/avatar/avatar2.png',
        text: $("#msg").val(),
        ago : ''
        };
        position = 'right';
        htmldiv = `<div class="answer ${position}">
                    <div class="avatar">
                      <img src="${ms.avatar}" alt="${ms.name}">
                      <div class="status online"></div>
                    </div>
                    <div class="text">
                      ${ms.text}
                    </div>
                    <div class="time">`+formatAMPM(new Date)+`</div>
                  </div>`;

        $( "div#chat-messages" ).append(htmldiv);
        $("#chat-messages").animate({ scrollTop: $('#chat-messages').prop("scrollHeight")}, 1000);
        
        $.post("/chat/"+$("#bot_id").text(),{"ques":$("#msg").val(),"prev_q":prev_q,"prev_a":prev_a},function(data, status){
          prev_a = data;
          prev_q = $("#msg").val();
          var ms = {
          username:'bot',
          name: '$("#bot_id").text()',
          avatar: bot_im_url,
          text: prev_a,
          ago : ''
          };
          position = 'left';
          htmldiv = `<div class="answer ${position}">
                      <div class="avatar">
                        <img src="${ms.avatar}" alt="${ms.name}">
                        <div class="status online"></div>
                      </div>
                      <div class="text">
                        ${ms.text}
                      </div>
                      <div class="time">`+formatAMPM(new Date)+`</div>
                    </div>`;

          $( "div#chat-messages" ).append(htmldiv);
          $("#chat-messages").animate({ scrollTop: $('#chat-messages').prop("scrollHeight")}, 1000);
          
          $("#msg").val("");
        });

    }

});